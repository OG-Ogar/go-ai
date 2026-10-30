# og-ai — a tiny cybersecurity language model, built from scratch

This repo trains a small **GPT-style language model from scratch** on a
cybersecurity corpus, using only PyTorch on a CPU. Every important line is
commented for a beginner, so the repo doubles as a guided tour of how a
language model actually works.

> **Honest expectations first.** This is a **1.6 million parameter** model
> trained for ~27 minutes on 1.28 MB of text on a laptop CPU. It learns the
> *shape* of cybersecurity writing — vocabulary, grammar, common phrases — and
> produces readable, on-topic paragraphs. It is **not** a reliable assistant,
> it does **not** contain "all knowledge of cybersecurity", and its output
> must never be trusted as fact. It is a teaching model, not a product.

---

## What is in the repo

```
og-ai/
├── data/
│   ├── notes/                  # hand-written cybersecurity notes (the "syllabus")
│   │   ├── 01_foundations.txt
│   │   ├── 02_threats_and_actors.txt
│   │   ├── 03_attacks.txt
│   │   ├── 04_cryptography.txt
│   │   ├── 05_network_and_endpoint.txt
│   │   ├── 06_identity_and_cloud.txt
│   │   ├── 07_operations_and_governance.txt
│   │   └── 08_qa_pairs.txt
│   └── cybersecurity_corpus.txt   # generated: notes + Wikipedia, one big file
├── scripts/
│   ├── build_corpus.py         # merges the notes with Wikipedia extracts
│   └── smoke_test.py           # fast "does everything connect?" test
├── src/
│   ├── config.py               # every hyperparameter in one place
│   ├── tokenizer.py            # text <-> integer ids (word level)
│   ├── dataset.py              # cuts the corpus into (input, next-token) pairs
│   ├── model.py                # the GPT itself  <-- the main learning file
│   ├── train.py                # the training loop
│   └── generate.py             # sample text from a trained checkpoint
├── checkpoints/                # created by training: model + tokenizer
└── requirements.txt
```

### Read the files in this order

| Order | File | What you learn |
|-------|------|----------------|
| 1 | `src/config.py` | What a hyperparameter is |
| 2 | `src/tokenizer.py` | How text becomes numbers (and back) |
| 3 | `src/dataset.py` | How "predict the next token" becomes training data |
| 4 | `src/model.py` | Attention, MLPs, residuals, embeddings — the whole GPT |
| 5 | `src/train.py` | Gradient descent, learning-rate schedule, checkpoints |
| 6 | `src/generate.py` | Sampling (temperature, top-k) |

---

## Setup

The project uses a virtual environment because modern Python blocks
system-wide installs.

```bash
cd og-ai
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs the **CPU** build of PyTorch, which is what this
machine can use (there is no GPU).

## Step 0 — quick self-test

```bash
.venv/bin/python scripts/smoke_test.py
```

This checks the tokenizer, the data loader, a forward/backward pass and
generation in a few seconds. Do this before a long training run.

## Step 1 — build the corpus

```bash
.venv/bin/python scripts/build_corpus.py
```

This reads `data/notes/*.txt`, downloads a curated set of Wikipedia articles
(CC BY-SA 4.0), and writes `data/cybersecurity_corpus.txt`. Wikipedia rate
limits aggressive clients, so the script paces itself and retries; a full run
takes a few minutes.

Use `--offline` to skip Wikipedia and train only on the local notes:

```bash
.venv/bin/python scripts/build_corpus.py --offline
```

## Step 2 — train

```bash
.venv/bin/python -m src.train
```

You will see losses like this:

```
step     0 | lr 3.00e-06 | train 8.71 | val 8.72 | ppl 6123 |   6.1s
step   100 | lr 3.00e-04 | train 5.10 | val 5.03 | ppl  153 |  52.9s
...
```

* **loss** — how surprised the model is by the correct next token.
  Lower is better. Blind guessing among 6000 tokens gives `ln(6000) ≈ 8.7`.
* **ppl (perplexity)** — `exp(loss)`; "confused between about N tokens".
* **val** — loss on text the model never trained on. Watch it: if `train`
  keeps falling while `val` rises, the model is memorising (overfitting).
  Training saves the checkpoint with the **best validation loss**, so you get
  the best generalising version, not the final one.

Checkpoints go to `checkpoints/`:
* `best.pt` — best validation loss
* `last.pt` — final step
* `tokenizer.json` — the 6000-token vocabulary

## Step 3 — generate

```bash
.venv/bin/python -m src.generate --prompt "A firewall is " --max-new-tokens 200
```

Interactive mode:

```bash
.venv/bin/python -m src.generate --interactive
```

Knobs worth trying:

```bash
--temperature 0.5   # safer, more repetitive
--temperature 1.0   # more random, more mistakes
--top-k 10          # only the 10 likeliest tokens
```

---

## How the model works (short version)

1. **Tokenize.** Text is split into words/punctuation; each gets an integer id.
2. **Embed.** Each id becomes a vector; a second table adds *position* info.
3. **Attention.** Every position asks "which earlier positions matter to me?"
   and blends their information. A causal mask stops it seeing the future.
4. **MLP.** Each position is processed independently through a small network.
5. **Repeat** steps 3–4 in a stack of blocks.
6. **Predict.** The final vector is projected to one score per token.
7. **Train.** Cross-entropy compares those scores to the real next token;
   backpropagation nudges every weight to be less wrong.
8. **Generate.** Sample a token, append it, repeat.

The maths that makes step 3 work is `softmax(QKᵀ / √d) · V`. The comments in
`src/model.py` explain each piece, including *why* the mask, the scaling, and
the residual connections exist.

---

## Tuning it

Everything lives in `src/config.py`.

| Want | Change |
|------|--------|
| Better quality | raise `n_layer`, `n_embd`, `block_size`, `max_steps` |
| Faster training | lower `n_layer`, `block_size`, `batch_size` |
| Less overfitting | raise `dropout`, add more text, lower `max_steps` |
| Smoother learning | lower `learning_rate` |
| Fewer `<unk>` tokens | raise `vocab_size` (costs memory and speed) |

On this machine (2 CPU cores, ~3.7 GB RAM) the defaults are deliberately sized
so a full run finishes in under half an hour. A bigger model is a one-line
change but will be much slower.

## Licence / sources

* Hand-written notes: this repository.
* Wikipedia extracts: CC BY-SA 4.0, attributed inside the generated corpus.
* Code: use it however you like.
