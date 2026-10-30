"""
smoke_test.py — prove the pipeline works before spending 15 minutes training.

Run:
    .venv/bin/python scripts/smoke_test.py

Each check prints PASS or FAIL. If everything passes, the model, tokenizer and
data loader are wired together correctly and training is safe to start.
"""

import os
import sys

import torch

# Make `import src...` work no matter where the script is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import Config            # noqa: E402
from src.dataset import encode_text, get_batch, load_corpus, train_val_split  # noqa: E402
from src.model import GPT                # noqa: E402
from src.tokenizer import WordTokenizer  # noqa: E402


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    # Returning a bool lets main() aggregate the result.
    return bool(condition)


def main():
    ok = True
    torch.manual_seed(0)
    torch.set_num_threads(2)

    # 1. Tokenizer round-trip: encode then decode must give the same text.
    tok = WordTokenizer.from_text("hello cybersecurity, 123.", max_vocab=100)
    sample = "hello cybersecurity, 123."
    ok &= check("tokenizer round-trip", tok.decode(tok.encode(sample)) == sample)
    ok &= check("vocabulary is non-trivial", tok.vocab_size >= 3)

    # 2. Corpus loads and is big enough to learn from.
    cfg = Config()
    text = load_corpus(os.path.join(ROOT, cfg.corpus_path))
    ok &= check(f"corpus loaded ({len(text):,} chars)", len(text) > 10_000)

    # 3. Batch shapes: x and y must be the same shape, y shifted by one.
    full_tok = WordTokenizer.from_text(text, max_vocab=cfg.vocab_size)
    data = encode_text(full_tok, train_val_split(text)[0])
    x, y = get_batch(data, cfg.block_size, 4)
    ok &= check("batch x shape", tuple(x.shape) == (4, cfg.block_size))
    ok &= check("batch y shape", tuple(y.shape) == (4, cfg.block_size))

    # 4. Forward + backward pass actually produces a finite loss and gradients.
    model = GPT(cfg, full_tok.vocab_size)
    logits, loss = model(x, y)
    loss.backward()
    ok &= check("logits shape", tuple(logits.shape) == (4, cfg.block_size, full_tok.vocab_size))
    ok &= check("loss is finite", torch.isfinite(loss).item())
    ok &= check("gradients exist",
                any(p.grad is not None for p in model.parameters()))

    # 5. Generation returns exactly the requested extra tokens.
    prompt = torch.zeros((1, 5), dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=5)
    ok &= check("generate adds 10 tokens", out.shape == (1, 15))

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
