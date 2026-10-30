"""
dataset.py — slicing the corpus into training examples.

BEGINNER NOTE
-------------
The model learns one simple game:

    "Here are N tokens. What is the NEXT token?"

So a single piece of training text like

    "The firewall blocks"

can be cut into many (input, answer) pairs ON THE CHARACTER LEVEL:

    "The fire"  -> "w"
    "The firew" -> "a"
    ...

In practice we do not store all of those. We take a random window of
`block_size + 1` tokens and split it:

    x = text[i : i + block_size]       <- what the model sees
    y = text[i + 1 : i + block_size + 1] <- the same text shifted by one

Then `x[t]` is the input and `y[t]` is the correct next token. This
shift-by-one trick is the whole reason a language model can be trained on
plain text with no human labelling.
"""

import torch  # tensors: the array type PyTorch works with


def load_corpus(path):
    """Read the training text from disk as one big Python string."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # A completely empty file would make the model impossible to train.
    if len(text) < 1000:
        raise ValueError(
            f"Corpus at {path} is only {len(text)} characters. "
            "Run scripts/build_corpus.py first."
        )
    return text


def train_val_split(text, val_fraction=0.1):
    """Split text into a training part and a held-out validation part.

    We never train on the validation part. It is our honest exam: if the
    validation loss goes UP while training loss goes DOWN, the model is
    memorising instead of learning (overfitting).
    """
    split = int(len(text) * (1.0 - val_fraction))  # 90% boundary
    return text[:split], text[split:]


def encode_text(tokenizer, text):
    """Text -> torch tensor of integer ids (int64 = 'long', required by
    nn.Embedding and F.cross_entropy)."""
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, block_size, batch_size, generator=None):
    """Sample one training batch.

    `data` is the full id tensor. We pick `batch_size` random starting
    positions and cut windows of length `block_size + 1` from each.
    """
    # Highest valid start index: we need block_size+1 tokens from it,
    # so we must stop before `len(data) - block_size - 1`.
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Text is shorter than block_size; lower block_size.")

    # torch.randint(low, high, size) samples integers in [low, high).
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)

    # Stack the windows into a 2D tensor: shape (batch_size, block_size + 1).
    window = torch.stack([data[s : s + block_size + 1] for s in starts])

    # x is everything except the last column, y is everything except the
    # first column. y is x shifted one token into the future.
    x = window[:, :-1]  # shape: (batch_size, block_size)
    y = window[:, 1:]   # shape: (batch_size, block_size)
    return x, y
