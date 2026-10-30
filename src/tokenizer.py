"""
tokenizer.py — turning text into numbers and back.

BEGINNER NOTE
-------------
Neural networks cannot read letters. They only do maths on numbers.
So the first job is a translation layer:

        "the firewall blocks traffic"
        ->  [17, 892, 3310, 45]

This project uses a WORD-level tokenizer:
    * split the text into words, numbers, punctuation marks and newlines
    * every distinct token gets an integer id
    * only the most common N tokens get their own id; anything rarer is
      mapped to a single special token called <unk> ("unknown")

Why word level instead of character level?
    * One token carries a whole word, so the model learns grammar and meaning
      from far fewer examples. On a CPU that difference is huge.
    * The trade-off is that very rare words become <unk>, and we must put the
      spaces back when converting ids to text (see detokenize below).
"""

import json
import re
from collections import Counter

# The splitting rule, read left to right:
#   \n            -> a newline is its own token (keeps paragraph structure)
#   [A-Za-z]+     -> runs of letters ("firewall")
#   [0-9]+        -> runs of digits ("2024", "443")
#   [^\sA-Za-z0-9] -> any other single non-space character ("." "," "-")
# Tokens are separated by whitespace, which we drop: it is reconstructed on
# the way back out, so storing it would waste vocabulary slots.
TOKEN_PATTERN = re.compile(r"\n|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")

# Detokenization rules: punctuation that hugs the token before it,
# and opening characters that hug the token after it.
NO_SPACE_BEFORE = set(".,;:!?)]}%'\"-/")
NO_SPACE_AFTER = set("([{$'\"-/.")


class WordTokenizer:
    """Maps tokens (words / punctuation / newlines) to integers and back."""

    UNK = "<unk>"  # id 0 by convention: "I have never seen this token"

    def __init__(self, tokens):
        # `tokens` is the ordered vocabulary. We keep it as a list so the
        # JSON file on disk is small and human-readable.
        self.tokens = list(tokens)

        # Guarantee <unk> exists and sits at index 0.
        if not self.tokens or self.tokens[0] != self.UNK:
            self.tokens = [self.UNK] + [t for t in self.tokens if t != self.UNK]

        # stoi = "string to integer": {"the": 17, ...}
        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}

        # itos = "integer to string": {17: "the", ...}
        self.itos = {i: tok for i, tok in enumerate(self.tokens)}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text, max_vocab=6000):
        """Build a vocabulary from the most common tokens in `text`.

        `max_vocab` caps the size. A bigger vocabulary keeps more rare words
        but makes the model's output layer heavier; 6000 covers ~95% of this
        corpus while staying cheap to train.
        """
        # findall() returns every token in order; Counter counts them.
        counts = Counter(TOKEN_PATTERN.findall(text))

        # most_common() sorts by frequency, highest first.
        # We reserve one slot for <unk>, hence max_vocab - 1.
        vocabulary = [tok for tok, _ in counts.most_common(max_vocab - 1)]
        return cls([cls.UNK] + vocabulary)

    # ------------------------------------------------------------------
    # The two functions that matter
    # ------------------------------------------------------------------

    def encode(self, text):
        """Text -> list of integer ids (one id per token)."""
        # .get(token, 0) falls back to <unk> for unseen tokens instead of
        # crashing, which matters because users type anything.
        return [self.stoi.get(tok, 0) for tok in TOKEN_PATTERN.findall(text)]

    def decode(self, ids):
        """List of integer ids -> text, with spaces re-inserted."""
        return self.detokenize([self.itos[int(i)] for i in ids])

    @staticmethod
    def detokenize(tokens):
        """Glue tokens back into readable text.

        The rules are simple English typography:
            "the" + "."       -> "the."      (no space before punctuation)
            "(" + "text"      -> "(text"     (no space after an opener)
            "don" + "'" + "t" -> "don't"     (apostrophes join both sides)
        """
        out = ""
        previous = None

        for tok in tokens:
            # Show unknown tokens as "?" rather than leaking "<unk>".
            if tok == WordTokenizer.UNK:
                tok = "?"

            # A newline always ends the current line.
            if tok == "\n":
                out = out.rstrip(" ") + "\n"
                previous = tok
                continue

            # Start of text, or start of a new line: no leading space.
            if not out or out.endswith("\n"):
                out += tok
            # Closing punctuation hugs the previous token.
            elif tok in NO_SPACE_BEFORE:
                out += tok
            # Opening punctuation means the next token hugs this one.
            elif out[-1] in NO_SPACE_AFTER:
                out += tok
            # Thousands separators and decimals: "1" "," "200" -> "1,200",
            # and "3" "." "14" -> "3.14". Only when both sides are digits.
            elif tok[0].isdigit() and previous in (",", ".") and len(out) > 1 and out[-2].isdigit():
                out += tok
            # Otherwise separate normal words with a single space.
            else:
                out += " " + tok

            previous = tok

        return out

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------

    @property
    def vocab_size(self):
        """How many distinct tokens exist (the model's output classes)."""
        return len(self.tokens)

    def save(self, path):
        """Write the vocabulary to disk so generation uses the same ids."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Rebuild a tokenizer from a file written by .save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["tokens"])
