"""
model.py — a small GPT (Generative Pre-trained Transformer) from scratch.

============================================================================
WHAT IS A GPT, IN ONE PARAGRAPH?
============================================================================
A GPT is a stack of identical "Transformer blocks". Each block does two
things to every token position in the text:

    1. ATTENTION: let this position look at earlier positions and pull in
       information it finds relevant.
    2. MLP: a small feed-forward network that thinks about what was gathered.

The output of the final block is turned into one score per possible next
token. Training pushes the correct next token's score up and all
the others down. That is the entire model.

============================================================================
SHAPES USED EVERYWHERE (memorise these three letters)
============================================================================
    B = batch size        how many independent text chunks at once
    T = time / sequence   how many tokens per chunk (= block_size)
    C = channels          the vector size per token (= n_embd)

So a tensor `x` of shape (B, T, C) means:
    "B separate texts, each T tokens long, each token stored as a
     C-dimensional vector."
============================================================================
"""

import math  # math.sqrt: used to scale attention scores

import torch  # PyTorch: tensors + automatic differentiation
import torch.nn as nn  # nn.Module (base class), nn.Linear, nn.Embedding, ...
from torch.nn import functional as F  # F.softmax, F.gelu, F.cross_entropy


# ============================================================================
# 1. CAUSAL SELF-ATTENTION
# ============================================================================
class CausalSelfAttention(nn.Module):
    """The part of the block where tokens exchange information.

    "Self-attention" means the sequence attends to itself (not to some other
    input). "Causal" means a token may only look BACKWARDS, never at the
    future — otherwise predicting the next token would be cheating.
    """

    def __init__(self, cfg):
        # Always call the parent class constructor first. This registers the
        # module with PyTorch so its parameters get tracked and saved.
        super().__init__()

        # Every head needs an equal slice of the C channels, so this must divide.
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = cfg.n_embd  # total channels (C) — needed to split Q/K/V
        self.n_head = cfg.n_head  # how many attention heads run in parallel
        self.head_dim = cfg.n_embd // cfg.n_head  # channels per single head
        self.dropout = cfg.dropout  # probability used by the dropout layers

        # One Linear layer produces Q, K and V for ALL heads at once.
        # It maps C channels -> 3*C channels, then we split the result in three.
        #   Q = "query"  : what am I looking for?
        #   K = "key"    : what do I contain?
        #   V = "value"  : what do I actually pass on if chosen?
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)

        # After attention we mix the heads back together with another Linear.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        # Dropout right after the attention weights (regularisation).
        self.attn_dropout = nn.Dropout(cfg.dropout)

        # Dropout on the final output of this sub-layer.
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # The causal mask: a lower-triangular matrix of ones.
        #   [[1, 0, 0],
        #    [1, 1, 0],
        #    [1, 1, 1]]
        # Reading row i: position i may attend to positions 0..i, nothing later.
        # register_buffer = store it with the model, move it with the model,
        # but DO NOT train it (it is a constant, not a weight).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).bool(),
            persistent=False,  # do not bloat the checkpoint file
        )

    def forward(self, x):
        # x shape: (B, T, C) — see the banner at the top of the file.
        B, T, C = x.shape

        # 1. Project the input into queries, keys and values in one go.
        #    Result shape: (B, T, 3*C).
        qkv = self.c_attn(x)

        # 2. Split the last dimension into three equal chunks of size C.
        #    q, k, v are each (B, T, C).
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 3. Reshape (B, T, C) -> (B, n_head, T, head_dim).
        #    We are separating heads so each can attend independently.
        #    .transpose(1, 2) swaps the T and n_head axes.
        #    .view is free (no copy); transpose is a view too; .contiguous()
        #    is applied later only where a real memory layout is needed.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 4. Score every query against every key.
        #    (B, n_head, T, head_dim) @ (B, n_head, head_dim, T)
        #    -> (B, n_head, T, T) : "how much does position i care about j?"
        #    Matmul sums over head_dim, so we divide by sqrt(head_dim) to keep
        #    the numbers from growing with dimension and saturating softmax.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 5. Apply the causal mask. We slice [:T, :T] in case the current
        #    sequence is shorter than block_size.
        #    masked_fill puts -infinity where the mask is False (future
        #    positions). softmax(-inf) becomes exactly 0, i.e. "not allowed".
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))

        # 6. Turn scores into probabilities that sum to 1 along the last axis.
        att = F.softmax(att, dim=-1)

        # 7. Regularise the attention weights during training.
        att = self.attn_dropout(att)

        # 8. Weighted sum of the values: (B, n_head, T, T) @ (B, n_head, T, d)
        #    -> (B, n_head, T, head_dim). Each position gets a blend of the
        #    values it decided were relevant.
        y = att @ v

        # 9. Undo the head split: (B, n_head, T, head_dim) -> (B, T, C).
        #    .transpose swaps the axes back, .contiguous() makes the memory
        #    layout normal again so .view can flatten the two head axes,
        #    and .view merges (n_head, head_dim) into C.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 10. Mix heads with a learned Linear and apply dropout.
        y = self.resid_dropout(self.c_proj(y))

        # Same shape we started with: (B, T, C). Good, it plugs into a residual.
        return y


# ============================================================================
# 2. FEED-FORWARD NETWORK ("MLP")
# ============================================================================
class MLP(nn.Module):
    """The "thinking" part of the block.

    Attention moves information BETWEEN positions. The MLP processes each
    position INDEPENDENTLY and gives the model non-linear capacity.
    The hidden layer is 4x wider than C, which is the standard GPT ratio.
    """

    def __init__(self, cfg):
        super().__init__()
        # Expand C -> 4C
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        # Squash back 4C -> C
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # Dropout on the way out
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # GELU is a smooth ReLU-like activation; GPT-2 uses it.
        x = self.fc(x)          # (B, T, C) -> (B, T, 4C)
        x = F.gelu(x)           # non-linearity, applied element-wise
        x = self.proj(x)        # (B, T, 4C) -> (B, T, C)
        x = self.dropout(x)     # regularisation
        return x


# ============================================================================
# 3. ONE TRANSFORMER BLOCK
# ============================================================================
class Block(nn.Module):
    """Attention + MLP, each wrapped in a residual connection.

    Residual connection means: output = input + change.
    This lets gradients flow straight down the stack, which is what makes
    deep networks trainable at all.

    "Pre-LN" means LayerNorm happens BEFORE attention/MLP (GPT-2 style),
    which is more stable than the original 2017 "post-LN" ordering.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)   # normalise before attention
        self.attn = CausalSelfAttention(cfg)  # the attention sub-layer
        self.ln2 = nn.LayerNorm(cfg.n_embd)   # normalise before MLP
        self.mlp = MLP(cfg)                   # the feed-forward sub-layer

    def forward(self, x):
        # Residual + pre-norm, twice. Note the `x +` on both lines.
        x = x + self.attn(self.ln1(x))  # communicate between positions
        x = x + self.mlp(self.ln2(x))   # think about each position
        return x


# ============================================================================
# 4. THE FULL GPT MODEL
# ============================================================================
class GPT(nn.Module):
    """Embeddings -> N blocks -> final LayerNorm -> vocabulary scores."""

    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg                    # keep the config for generate()
        self.block_size = cfg.block_size  # max context length
        self.vocab_size = vocab_size      # number of possible tokens

        # Token embedding: maps each token id to a learned C-dim vector.
        # This is the "meaning" table the model learns.
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)

        # Position embedding: a second learned table, one vector per slot
        # 0..block_size-1. Without it, attention is order-blind.
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        # Dropout after adding the two embeddings.
        self.drop = nn.Dropout(cfg.dropout)

        # The stack of identical blocks (each has its own weights).
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        # Final normalisation before the output layer.
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Output layer: C -> vocab_size scores ("logits") per position.
        # No bias needed for the output projection.
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        # WEIGHT TYING: reuse the token embedding matrix as the output matrix.
        # Saves parameters and usually improves quality, because "input
        # meaning of token X" and "score for token X" share statistics.
        self.lm_head.weight = self.tok_emb.weight

        # Initialise every weight sensibly before training starts.
        self.apply(self._init_weights)

        # GPT-2 trick: scale down the residual output projections by
        # 1/sqrt(2 * n_layer) so that the residual stream does not grow
        # larger and larger as we stack blocks.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        """Called by self.apply() on every sub-module."""
        if isinstance(module, nn.Linear):
            # Small random normal weights; zero bias.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Same small normal distribution for embedding tables.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Run the model.

        idx     : (B, T) integer token ids
        targets : (B, T) correct next ids, or None at generation time

        Returns (logits, loss):
            logits : (B, T, vocab_size) raw scores for every position
            loss   : a single number (or None if targets is None)
        """
        B, T = idx.shape  # unpack batch size and sequence length

        # Positions 0, 1, 2, ... T-1, placed on the same device as the input.
        pos = torch.arange(T, device=idx.device)

        # Look up embeddings and ADD them: (B,T,C) + (T,C) broadcasts the
        # position table across the batch.
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Regularise the embeddings.
        x = self.drop(x)

        # Push through every block in order.
        for block in self.blocks:
            x = block(x)

        # Final normalisation.
        x = self.ln_f(x)

        # Project to vocabulary scores: (B, T, C) -> (B, T, vocab_size).
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy expects a flat list of predictions and answers.
            # (B, T, V) -> (B*T, V) and (B, T) -> (B*T).
            #
            # We use .reshape(), not .view(). `view` only works on tensors
            # whose memory is laid out contiguously; `targets` came from a
            # slice like window[:, 1:] and is therefore non-contiguous.
            # reshape() copies when needed, so it always works.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,  # -1 would mean "do not score this position"
            )
        return logits, loss

    @torch.no_grad()  # generation does not need gradients: saves memory/time
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Sample new tokens one at a time.

        idx          : (B, T) starting prompt ids
        temperature  : <1 = safer/repetitive, >1 = wilder/creative
        top_k        : only sample from the k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Never feed more than block_size tokens (position table limit).
            idx_cond = idx[:, -self.block_size:]

            # Forward pass; we only need the logits.
            logits, _ = self(idx_cond)

            # Keep the scores for the LAST position only: that is the
            # prediction for the next token. -> (B, vocab_size)
            logits = logits[:, -1, :]

            # Temperature: divide scores before softmax. Lower T sharpens the
            # distribution, higher T flattens it.
            logits = logits / temperature

            # Top-k filtering: keep the k best scores, set the rest to -inf,
            # so softmax gives them zero probability.
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                # v[:, [-1]] is the k-th best value; keep a (B,1) shape so it
                # broadcasts against (B, vocab_size).
                logits[logits < values[:, [-1]]] = float("-inf")

            # Convert scores to probabilities.
            probs = F.softmax(logits, dim=-1)

            # Actually roll the dice. multinomial samples one id per batch row.
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append it to the sequence and loop again.
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    def num_parameters(self):
        """Count trainable numbers (a quick sanity check of model size)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95)):
        """Build AdamW with the standard GPT weight-decay rules."""
        # Split parameters into two groups:
        #   * decay: 2-D matrices (Linear / Embedding weights)
        #   * no_decay: biases and LayerNorm scales
        # Applying weight decay to biases/LayerNorm hurts training.
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # AdamW is Adam with "decoupled" weight decay (the correct way to do it).
        # betas=(0.9, 0.95) are the values used by GPT-2/3.
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
