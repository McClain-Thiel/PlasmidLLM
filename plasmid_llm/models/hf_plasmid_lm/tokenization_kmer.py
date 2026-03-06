"""K-mer tokenizer for PlasmidLM — encodes DNA as overlapping k-mers."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from typing import List, Optional

from transformers import PreTrainedTokenizer

BASES = "ACGT"


def build_kmer_vocab(special_tokens: list[str], k: int = 6) -> dict[str, int]:
    """Build vocabulary: special tokens first, then all 4^k k-mers in lexicographic order."""
    vocab = {}
    for i, tok in enumerate(special_tokens):
        vocab[tok] = i
    next_id = len(vocab)
    for kmer in itertools.product(BASES, repeat=k):
        token = "".join(kmer)
        vocab[token] = next_id
        next_id += 1
    return vocab


class PlasmidKmerTokenizer(PreTrainedTokenizer):
    """Overlapping k-mer tokenizer for plasmid DNA sequences.

    DNA segments are split into k-mers with configurable stride (overlap = k - stride).
    Special tokens (<AMR_...>, <ORI_...>, etc.) are preserved as single tokens.
    N bases are replaced with a deterministic pseudo-random base seeded by position.

    Args:
        vocab_file: Path to vocab.json
        k: K-mer size (default 6)
        stride: Step size between k-mers (default 3, giving overlap of 3)
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        k: int = 6,
        stride: int = 3,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        sep_token: str = "<SEP>",
        **kwargs,
    ):
        self.k = k
        self.stride = stride

        with open(vocab_file, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "token_to_id" in data:
            data = data["token_to_id"]

        self._vocab = data
        self._id_to_token = {v: k for k, v in self._vocab.items()}

        # Only pass special tokens that exist in vocab
        special_kwargs = {}
        for name, tok in [("bos_token", bos_token), ("eos_token", eos_token),
                          ("unk_token", unk_token), ("pad_token", pad_token),
                          ("sep_token", sep_token)]:
            if tok in self._vocab:
                special_kwargs[name] = tok

        # Store k and stride in kwargs so they're saved/loaded
        kwargs["k"] = k
        kwargs["stride"] = stride
        # Disable automatic BOS/EOS insertion (transformers 5.x); prompts
        # already contain <BOS>...<SEP> and the model generates <EOS> itself.
        kwargs.setdefault("special_tokens_pattern", "none")
        super().__init__(**special_kwargs, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    @staticmethod
    def _clean_dna(seq: str) -> str:
        """Uppercase, strip whitespace, replace N with deterministic base."""
        seq = re.sub(r"\s+", "", seq).upper()
        result = []
        for i, c in enumerate(seq):
            if c == "N":
                # Deterministic replacement seeded by position
                h = int(hashlib.md5(str(i).encode()).hexdigest(), 16)
                result.append(BASES[h % 4])
            else:
                result.append(c)
        return "".join(result)

    def _kmerize(self, dna: str) -> list[str]:
        """Split DNA into overlapping k-mers with self.stride."""
        k, stride = self.k, self.stride
        if len(dna) < k:
            # Tail shorter than k: pad with A to make a full k-mer
            return [dna.ljust(k, "A")] if dna else []
        kmers = []
        for i in range(0, len(dna) - k + 1, stride):
            kmers.append(dna[i:i + k])
        # Handle tail: if the last k-mer doesn't reach the end, add one more
        last_start = (len(kmers) - 1) * stride
        if last_start + k < len(dna):
            tail = dna[last_start + stride:]
            if len(tail) < k:
                tail = tail.ljust(k, "A")
            kmers.append(tail)
        return kmers

    def _tokenize(self, text: str) -> List[str]:
        """Split text into special tokens and k-merized DNA segments."""
        # Split on special tokens <...>
        parts = re.split(r"(<[^>]+>)", text)
        tokens = []
        for part in parts:
            if not part or part.isspace():
                continue
            if part.startswith("<") and part.endswith(">"):
                tokens.append(part)
            else:
                dna = self._clean_dna(part)
                if dna:
                    tokens.extend(self._kmerize(dna))
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get("<UNK>", 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "<UNK>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Reconstruct text from k-mer tokens.

        For consecutive DNA k-mers, take the first `stride` chars from each
        except the last (take all k). Special tokens are kept as-is.
        """
        result = []
        dna_run: list[str] = []

        def flush_dna():
            if not dna_run:
                return
            if len(dna_run) == 1:
                result.append(dna_run[0])
            else:
                parts = [kmer[:self.stride] for kmer in dna_run[:-1]]
                parts.append(dna_run[-1])  # last k-mer: take all
                result.append("".join(parts))
            dna_run.clear()

        for tok in tokens:
            if tok.startswith("<") and tok.endswith(">"):
                flush_dna()
                result.append(tok)
            else:
                dna_run.append(tok)
        flush_dna()
        return "".join(result)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        with open(vocab_file, "w") as f:
            json.dump(self._vocab, f, indent=2)
        return (vocab_file,)
