
import regex as re
from collections import Counter
from pathlib import Path


_GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


_BYTE_TOKENS = tuple(bytes([byte_val]) for byte_val in range(256))


def _split_on_special_tokens(text: bytes, special_tokens: list[bytes]) -> list[bytes]:
    if not special_tokens:
        return [text]

    parts: list[bytes] = []
    pos = 0
    while pos < len(text):
        next_pos = len(text)
        next_token_len = 0
        for special_token in special_tokens:
            idx = text.find(special_token, pos)
            if idx != -1 and idx < next_pos:
                next_pos = idx
                next_token_len = len(special_token)

        if next_pos > pos:
            parts.append(text[pos:next_pos])

        if next_pos == len(text):
            break

        pos = next_pos + next_token_len

    return parts


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    first, second = pair
    merged: list[bytes] = []
    i = 0
    word_len = len(word)
    while i < word_len:
        if i < word_len - 1 and word[i] == first and word[i + 1] == second:
            merged.append(first + second)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def _word_has_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> bool:
    first, second = pair
    for i in range(len(word) - 1):
        if word[i] == first and word[i + 1] == second:
            return True
    return False


def _add_word_pairs(pair_counts: Counter, word: tuple[bytes, ...], freq: int) -> None:
    for i in range(len(word) - 1):
        pair_counts[(word[i], word[i + 1])] += freq


def _remove_word_pairs(pair_counts: Counter, word: tuple[bytes, ...], freq: int) -> None:
    for i in range(len(word) - 1):
        pair_counts[(word[i], word[i + 1])] -= freq

def train_bpe(input_path: Path, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:  
    with open(input_path, "rb") as f:
        text = f.read()

        special_bytes = [token.encode("utf-8") for token in special_tokens]
        special_bytes_set = set(special_bytes)

        vocab: dict[int, bytes] = {i: token_bytes for i, token_bytes in enumerate(special_bytes)}
        next_id = len(vocab)

        word_freqs: dict[tuple[bytes, ...], int] = {}
        for chunk in _split_on_special_tokens(text, special_bytes):
            chunk_text = chunk.decode("utf-8", errors="strict")
            for pretoken in _GPT2_PRETOKENIZE_PATTERN.findall(chunk_text):
                word = tuple(_BYTE_TOKENS[byte_val] for byte_val in pretoken.encode("utf-8"))
                word_freqs[word] = word_freqs.get(word, 0) + 1

        for byte_val in range(256):
            byte_token = _BYTE_TOKENS[byte_val]
            if byte_token not in special_bytes_set:
                vocab[next_id] = byte_token
                next_id += 1

        merges: list[tuple[bytes, bytes]] = []
        pair_counts: Counter = Counter()
        for word, freq in word_freqs.items():
            _add_word_pairs(pair_counts, word, freq)

        while len(vocab) < vocab_size:
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            if pair_counts[best_pair] <= 0:
                break

            merges.append(best_pair)
            vocab[next_id] = best_pair[0] + best_pair[1]
            next_id += 1

            words_to_update = [word for word in word_freqs if _word_has_pair(word, best_pair)]
            for word in words_to_update:
                freq = word_freqs.pop(word)
                _remove_word_pairs(pair_counts, word, freq)

                merged_word = _merge_word(word, best_pair)
                merged_freq = word_freqs.pop(merged_word, 0) + freq
                word_freqs[merged_word] = merged_freq
                _add_word_pairs(pair_counts, merged_word, merged_freq)

        return vocab, merges