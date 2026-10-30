"""
generate.py — use the trained model to produce text.

BEGINNER NOTE
-------------
Generation is the training game played forwards:

    give the model some tokens  ->  it predicts the next one
    append that token           ->  predict the next one again
    ... repeat N times

This is called "autoregressive" generation because each output becomes part
of the next input.

Run it:
    .venv/bin/python -m src.generate --prompt "Phishing is "
    .venv/bin/python -m src.generate --interactive
"""

import argparse  # clean command-line flags without manual string parsing
import os

import torch

from .config import Config
from .model import GPT
from .tokenizer import WordTokenizer


def load_model(checkpoint_path, device="cpu"):
    """Rebuild the exact model + tokenizer that was saved during training."""
    # weights_only=False because our checkpoint holds plain Python data
    # (config dict, vocabulary list) next to the tensors.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Recreate the Config from the saved dict. Unknown keys are ignored in
    # case an older checkpoint lacks a newer field.
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in Config.__annotations__})

    # Recreate the tokenizer from its saved vocabulary.
    tokenizer = WordTokenizer(ckpt["vocab"])

    # Recreate the model with the SAME architecture, then load the weights.
    model = GPT(cfg, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # inference mode: dropout off

    return model, tokenizer, cfg, ckpt


def generate_text(model, tokenizer, cfg, prompt, max_new_tokens=400,
                  temperature=0.8, top_k=40):
    """Encode the prompt, sample, decode the whole sequence back to text."""
    # encode() maps words the model never saw to <unk> for us, so there is
    # nothing to filter here. We only need at least one token to condition on.
    if not prompt.strip():
        prompt = "\n"

    # (1, T) tensor, because the model always expects a batch dimension.
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    # Ask the model for new tokens.
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # out is (1, T + new); decode row 0 to a string.
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Sample text from the cybersecurity GPT")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt", default="A firewall is ")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="<1 = conservative, >1 = more random")
    parser.add_argument("--top-k", type=int, default=40,
                        help="sample only from the k most likely tokens")
    parser.add_argument("--interactive", action="store_true",
                        help="read prompts from the keyboard in a loop")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. Train first: "
            ".venv/bin/python -m src.train"
        )

    model, tokenizer, cfg, ckpt = load_model(args.checkpoint)
    print(f"Loaded {args.checkpoint} (step {ckpt['step']}, "
          f"val loss {ckpt['val_loss']:.3f})")

    if args.interactive:
        print("Type a prompt and press Enter. Ctrl+C to quit.")
        while True:
            try:
                prompt = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            print(generate_text(model, tokenizer, cfg, prompt,
                                args.max_new_tokens, args.temperature, args.top_k))
    else:
        print("-" * 70)
        print(generate_text(model, tokenizer, cfg, args.prompt,
                            args.max_new_tokens, args.temperature, args.top_k))
        print("-" * 70)


if __name__ == "__main__":
    main()
