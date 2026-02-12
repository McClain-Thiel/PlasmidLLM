"""WordLevel tokenizer for plasmid sequences with special tokens and DNA bases."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers


# DNA bases to add to vocab (upper + lower)
DNA_BASES = list("ATCGNatcgn")

# Regex that splits on angle-bracket tokens while preserving them
_TAG_PATTERN = re.compile(r"(<[^>]+>)")


def _load_vocab(path: str | Path) -> dict[str, int]:
    """Load token vocabulary from JSON file. Supports local paths and S3 URIs."""
    path_str = str(path)
    if path_str.startswith("s3://"):
        import boto3

        parts = path_str.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        vocab = json.loads(obj["Body"].read().decode("utf-8"))
    else:
        with open(path) as f:
            vocab = json.load(f)
    return vocab


class PlasmidTokenizer:
    """WordLevel tokenizer for plasmid tag+DNA sequences.

    Vocabulary consists of special tokens (like <BOS>, <AMR_KANAMYCIN>, <SEP>)
    plus individual DNA base characters (A, T, C, G, N and lowercase).
    """

    def __init__(self, vocab_path: str | Path):
        raw_vocab = _load_vocab(vocab_path)

        # Ensure DNA bases are in the vocab
        next_id = max(raw_vocab.values()) + 1 if raw_vocab else 0
        for base in DNA_BASES:
            if base not in raw_vocab:
                raw_vocab[base] = next_id
                next_id += 1

        self._vocab = raw_vocab
        self._id_to_token = {v: k for k, v in raw_vocab.items()}

        # Build tokenizers-library tokenizer for fast encoding
        self._tokenizer = Tokenizer(models.WordLevel(vocab=raw_vocab, unk_token="<UNK>"))
        self._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
            _TagSplitter()
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        return self._vocab.get("<PAD>", 0)

    @property
    def bos_token_id(self) -> int:
        return self._vocab.get("<BOS>", 1)

    @property
    def eos_token_id(self) -> int:
        return self._vocab.get("<EOS>", 2)

    @property
    def sep_token_id(self) -> int:
        return self._vocab.get("<SEP>", 3)

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs using regex-based splitting."""
        tokens = _split_tokens(text)
        ids = []
        unk_id = self._vocab.get("<UNK>", 0)
        for tok in tokens:
            ids.append(self._vocab.get(tok, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for i in ids:
            tok = self._id_to_token.get(i, "<UNK>")
            tokens.append(tok)
        return "".join(tokens)


def _split_tokens(text: str) -> list[str]:
    """Split text into special tokens and individual characters.

    '<BOS><AMR_KAN>ATCG' -> ['<BOS>', '<AMR_KAN>', 'A', 'T', 'C', 'G']
    """
    parts = _TAG_PATTERN.split(text)
    tokens = []
    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            tokens.append(part)
        else:
            tokens.extend(list(part))
    return tokens


class _TagSplitter:
    """Custom pre-tokenizer for the tokenizers library."""

    def split(self, _i: int, normalized: str) -> list[tuple[str, tuple[int, int]]]:
        tokens = _split_tokens(normalized)
        result = []
        offset = 0
        for tok in tokens:
            start = normalized.find(tok, offset)
            if start == -1:
                start = offset
            end = start + len(tok)
            result.append((tok, (start, end)))
            offset = end
        return result

    def pre_tokenize(self, pretok):
        pretok.split(self.split)
