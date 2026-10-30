"""
config.py — every "knob" of the project in one place.

BEGINNER NOTE
-------------
A neural network has many settings (called "hyperparameters"). Instead of
scattering numbers all over the code, we put them in ONE object. If you want
a bigger/smaller model, or a faster/slower training run, change a number
HERE and nothing else.

A "dataclass" is just a Python class that automatically creates
__init__, __repr__ etc. for you from the type-annotated fields below.
`Config()` therefore builds a config object with all the default values.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 1. MODEL SHAPE — how big the brain is
    # ------------------------------------------------------------------

    # block_size = how many tokens the model can look at at once.
    # The model is trained to predict the NEXT token given at most this
    # many previous tokens. Bigger = more context, but much slower.
    # (Measured on this 2-core CPU: 192 made a run take ~57 min.)
    block_size: int = 128

    # n_layer = how many Transformer "blocks" to stack.
    # More blocks = deeper reasoning, more parameters, slower training.
    n_layer: int = 4

    # n_head = how many attention "heads" run in parallel inside each block.
    # Each head can learn to look at a different kind of relationship.
    # Must divide n_embd evenly.
    n_head: int = 4

    # n_embd = size of the vector that represents each token.
    # This is the main "width" of the network. Must be divisible by n_head.
    n_embd: int = 128

    # dropout = randomly turn off this fraction of neurons during training.
    # This is a regularisation trick that stops the model memorising the
    # training text. 0.0 = no dropout, 0.1 = drop 10%, etc.
    dropout: float = 0.1

    # bias = should Linear layers include a learnable "+ b" offset?
    # Modern GPT models usually keep this True for simplicity.
    bias: bool = True

    # vocab_size = how many distinct tokens the tokenizer keeps.
    # 6000 covers about 95% of this corpus; rarer words become <unk>.
    # A bigger vocabulary means a bigger (slower) output layer.
    vocab_size: int = 6000

    # ------------------------------------------------------------------
    # 2. TRAINING — how the model learns
    # ------------------------------------------------------------------

    # batch_size = how many independent text chunks we look at per step.
    # Bigger = smoother learning but more RAM. On 3.7 GB RAM keep this small.
    batch_size: int = 8

    # learning_rate = how big a step the optimiser takes.
    # Too big: loss explodes. Too small: training crawls.
    learning_rate: float = 3e-4

    # max_steps = total number of optimiser steps to run.
    # ~0.8 s/step on this machine, so 2000 steps is roughly 27 minutes.
    max_steps: int = 2000

    # warmup_steps = slowly ramp the learning rate up at the start.
    # Starting cold at full speed can wreck the random initial weights.
    warmup_steps: int = 100

    # min_lr_ratio = at the end of training the LR decays to
    # learning_rate * min_lr_ratio (cosine schedule). Small but > 0.
    min_lr_ratio: float = 0.1

    # weight_decay = mild pull that keeps weights small (regularisation).
    # Applied only to big matrix weights, not to biases / LayerNorms.
    weight_decay: float = 0.01

    # grad_clip = cap the size of gradients so one bad batch cannot blow up
    # the whole network ("gradient clipping").
    grad_clip: float = 1.0

    # eval_interval = every N steps, measure validation loss.
    eval_interval: int = 100

    # eval_iters = how many batches to average when measuring loss.
    # More = more accurate but slower evaluation.
    eval_iters: int = 20

    # seed = fixed random seed so runs are reproducible.
    seed: int = 1337

    # ------------------------------------------------------------------
    # 3. FILES
    # ------------------------------------------------------------------

    # Where the training text lives.
    corpus_path: str = "data/cybersecurity_corpus.txt"

    # Where model checkpoints + tokenizer are written.
    checkpoint_dir: str = "checkpoints"
