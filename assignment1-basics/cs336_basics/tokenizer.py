from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

from cs336_basics.bpe import _GPT2_PRETOKENIZE_PATTERN


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _bytes_to_unicode_str(token: bytes) -> str:
    encoder = gpt2_bytes_to_unicode()
    return "".join(encoder[b] for b in token)


def _unicode_str_to_bytes(token: str) -> bytes:
    decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return bytes(decoder[ch] for ch in token)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._max_special_prefix = max((len(token) for token in self.special_tokens), default=0)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(token_id): _unicode_str_to_bytes(token_str)
            for token_str, token_id in raw_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                left, right = line.split(" ")
                merges.append((_unicode_str_to_bytes(left), _unicode_str_to_bytes(right)))

        if special_tokens:
            for special_token in special_tokens:
                special_bytes = special_token.encode("utf-8")
                if special_bytes not in set(vocab.values()):
                    vocab[len(vocab)] = special_bytes

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def save(self, vocab_filepath: str | Path, merges_filepath: str | Path) -> None:
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        merges_path.parent.mkdir(parents=True, exist_ok=True)

        serializable_vocab = {
            _bytes_to_unicode_str(token_bytes): token_id
            for token_id, token_bytes in sorted(self.vocab.items())
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write(f"{_bytes_to_unicode_str(left)} {_bytes_to_unicode_str(right)}\n")

    def encode(self, text: str) -> list[int]:
        return list(self._encode_parts(text))

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            to_process, buffer = self._split_buffer_for_processing(buffer)
            if to_process:
                yield from self._encode_parts(to_process)

        if buffer:
            yield from self._encode_parts(buffer)

    def _split_buffer_for_processing(self, buffer: str) -> tuple[str, str]:
        if not self.special_tokens:
            return buffer, ""

        for hold_len in range(min(self._max_special_prefix, len(buffer)), 0, -1):
            suffix = buffer[-hold_len:]
            if any(
                special_token.startswith(suffix) and len(suffix) < len(special_token)
                for special_token in self.special_tokens
            ):
                return buffer[:-hold_len], suffix

        return buffer, ""

    def _encode_parts(self, text: str) -> Iterator[int]:
        pos = 0
        text_len = len(text)

        while pos < text_len:
            for special_token in self.special_tokens_sorted:
                if text.startswith(special_token, pos):
                    special_bytes = special_token.encode("utf-8")
                    yield self.bytes_to_id[special_bytes]
                    pos += len(special_token)
                    break
            else:
                next_special = text_len
                for special_token in self.special_tokens_sorted:
                    idx = text.find(special_token, pos)
                    if idx != -1 and idx < next_special:
                        next_special = idx

                chunk = text[pos:next_special]
                for pretoken in _GPT2_PRETOKENIZE_PATTERN.findall(chunk):
                    for token_bytes in self._apply_bpe(pretoken):
                        yield self.bytes_to_id[token_bytes]
                pos = next_special

    def _apply_bpe(self, pretoken: str) -> list[bytes]:
        word = [bytes([byte_val]) for byte_val in pretoken.encode("utf-8")]

        while len(word) >= 2:
            best_rank: int | None = None
            best_index: int | None = None

            for index in range(len(word) - 1):
                pair = (word[index], word[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index

            if best_index is None:
                break

            word[best_index] = word[best_index] + word[best_index + 1]
            del word[best_index + 1]

        return word
