#!/usr/bin/env python3
"""Train a BPE tokenizer and save vocab/merges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_bpe.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Training text path")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token (repeatable). Default: <|endoftext|>",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()

    # argparse append with default can duplicate; normalize
    special_tokens = list(dict.fromkeys(args.special_token))

    print(f"Training BPE on {args.input} (vocab_size={args.vocab_size})...")
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.out_dir / "vocab.json"
    merges_path = args.out_dir / "merges.txt"
    tokenizer.save(vocab_path, merges_path)
    print(f"Saved vocab ({len(vocab)} tokens) -> {vocab_path}")
    print(f"Saved merges ({len(merges)}) -> {merges_path}")


if __name__ == "__main__":
    main()
