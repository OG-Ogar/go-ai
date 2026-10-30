"""
train.py — the training loop.

BEGINNER NOTE
-------------
Training is a repeat-until-bored loop:

    1. Grab a random batch of text.
    2. Ask the model to predict the next token at every position.
    3. Measure how wrong it was (the "loss").
    4. Nudge every weight a tiny bit in the direction that lowers the loss.
    5. Repeat.

Step 4 is done by PyTorch's autograd + an optimiser (AdamW). You never write
the calculus yourself; you just call `loss.backward()` and `optimizer.step()`.

Run it with:
    .venv/bin/python -m src.train
"""

import argparse            # small command-line flags, e.g. --steps
import math                # exp() to turn loss into "perplexity"
import os                  # path joining, directory creation
import time                # timing how fast training runs
from dataclasses import asdict  # turn our Config object into a plain dict

import torch
from torch.nn import functional as F  # noqa: F401  (kept for experiments)

from .config import Config
from .dataset import encode_text, get_batch, load_corpus, train_val_split
from .model import GPT
from .tokenizer import WordTokenizer


def get_lr(step, cfg):
    """Learning-rate schedule: warm up, then cosine-decay down.

    * Warmup: start near 0 and climb to cfg.learning_rate over warmup_steps.
      Random initial weights can produce huge gradients; ramping up avoids
      an early blow-up.
    * Cosine decay: smoothly shrink to learning_rate * min_lr_ratio.
      Large steps early explore; small steps late fine-tune.
    """
    # Phase 1: linear warmup.
    if step < cfg.warmup_steps:
        # step+1 so the very first step is not exactly 0.
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    # Phase 2: cosine decay. `progress` goes 0 -> 1 over the remaining steps.
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)  # clamp in case we run past max_steps

    # Cosine goes 1 -> -1, so (1 + cos) / 2 goes 1 -> 0 smoothly.
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    # Blend between the minimum and the full learning rate.
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + cosine * (cfg.learning_rate - min_lr)


@torch.no_grad()  # evaluation must not build a gradient graph
def estimate_loss(model, data, cfg, generator=None):
    """Average loss over several random batches (a stable estimate)."""
    model.eval()  # turns OFF dropout, so results are deterministic

    losses = torch.zeros(cfg.eval_iters)  # one slot per batch
    for i in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.block_size, cfg.batch_size, generator)
        _, loss = model(x, y)
        losses[i] = loss.item()  # .item() -> plain Python float

    model.train()  # turn dropout back ON for further training
    return losses.mean().item()


def save_checkpoint(path, model, cfg, tokenizer, step, train_loss, val_loss):
    """Bundle everything needed to reproduce generation into one file."""
    torch.save(
        {
            # The learned weights (a dict of tensors).
            "model": model.state_dict(),
            # The architecture settings, so we can rebuild the same shape.
            "config": asdict(cfg),
            # The token vocabulary, so ids decode back to the same text.
            "vocab": tokenizer.tokens,
            "vocab_size": tokenizer.vocab_size,
            # Bookkeeping.
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    cfg = Config()

    # Small convenience override so you can test the pipeline quickly:
    #     .venv/bin/python -m src.train --steps 30
    parser = argparse.ArgumentParser(description="Train the cybersecurity GPT")
    parser.add_argument("--steps", type=int, default=None,
                        help="override cfg.max_steps (useful for a quick test)")
    args = parser.parse_args()
    if args.steps is not None:
        cfg.max_steps = args.steps

    # Reproducibility: same seed -> same random batches -> same result.
    torch.manual_seed(cfg.seed)

    # This machine has 2 CPU cores. Telling PyTorch avoids thread thrash.
    torch.set_num_threads(2)

    # We train on CPU because there is no GPU on this machine.
    device = "cpu"

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.corpus_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    text = load_corpus(cfg.corpus_path)
    print(f"Corpus: {len(text):,} characters")
    # Build the vocabulary from the corpus itself (word level).
    tokenizer = WordTokenizer.from_text(text, max_vocab=cfg.vocab_size)
    tokenizer.save(os.path.join(cfg.checkpoint_dir, "tokenizer.json"))
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    # 90% train / 10% validation, split at a single cut point.
    train_text, val_text = train_val_split(text, val_fraction=0.1)
    train_data = encode_text(tokenizer, train_text)
    val_data = encode_text(tokenizer, val_text)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    model = GPT(cfg, tokenizer.vocab_size).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val = float("inf")  # best validation loss seen so far
    t0 = time.time()         # start the stopwatch

    for step in range(cfg.max_steps):
        # Apply the schedule for THIS step.
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 1. One random batch of (inputs, correct next tokens).
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size)

        # 2. Forward pass: predictions + loss.
        _, loss = model(x, y)

        # 3. Backward pass: compute d(loss)/d(every weight).
        optimizer.zero_grad(set_to_none=True)  # clear old gradients
        loss.backward()

        # 4. Gradient clipping: rescale all gradients if their combined
        #    norm exceeds cfg.grad_clip. Prevents one bad batch from
        #    destroying the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # 5. Update every weight.
        optimizer.step()

        # ------------------------------------------------------------------
        # Logging / evaluation
        # ------------------------------------------------------------------
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            train_loss = estimate_loss(model, train_data, cfg)
            val_loss = estimate_loss(model, val_data, cfg)

            # Perplexity = exp(loss). Easier to interpret: "the model is about
            # as confused as choosing between this many equally likely chars".
            ppl = math.exp(val_loss)

            print(
                f"step {step:5d} | lr {lr:.2e} | "
                f"train {train_loss:.3f} | val {val_loss:.3f} | "
                f"ppl {ppl:7.2f} | {time.time() - t0:5.1f}s"
            )

            # Keep the checkpoint with the best validation loss.
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, "best.pt"),
                    model, cfg, tokenizer, step, train_loss, val_loss,
                )

    # Always save the final state too.
    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "last.pt"),
        model, cfg, tokenizer, cfg.max_steps - 1, train_loss, val_loss,
    )

    print(f"\nDone in {time.time() - t0:.1f}s. Best val loss: {best_val:.3f}")
    print(f"Checkpoints written to {cfg.checkpoint_dir}/")
    print("Next:  .venv/bin/python -m src.generate --prompt \"A firewall is \"")


if __name__ == "__main__":
    main()
