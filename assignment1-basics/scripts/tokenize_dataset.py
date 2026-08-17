#!/usr/bin/env python3
"""Encode a text corpus with a trained BPE tokenizer into a uint16 .npy / .bin file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw text corpus")
    parser.add_argument("--vocab", type=Path, required=True, help="vocab.json")
    parser.add_argument("--merges", type=Path, required=True, help="merges.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path")
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
    )
    args = parser.parse_args()
    special_tokens = list(dict.fromkeys(args.special_token))

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=special_tokens)
    assert len(tokenizer.vocab) <= np.iinfo(np.uint16).max + 1, "vocab too large for uint16"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")

    count = 0
    with open(args.input, encoding="utf-8") as f_in, open(tmp_path, "wb") as f_out:
        for token_id in tqdm(tokenizer.encode_iterable(f_in), desc="Encoding"):
            f_out.write(np.uint16(token_id).tobytes())
            count += 1

    tokens = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    assert tokens.shape[0] == count
    np.save(args.output, np.array(tokens, dtype=np.uint16))
    tmp_path.unlink()
    print(f"Wrote {count:,} tokens -> {args.output}")


if __name__ == "__main__":
    main()
