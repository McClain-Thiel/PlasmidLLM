"""HuggingFace-compatible tokenizer for PlasmidLM."""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from transformers import PreTrainedTokenizer


DNA_BASES = list("ATCGNatcgn")


class PlasmidLMTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer for plasmid sequences with special tokens."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        sep_token: str = "<SEP>",
        **kwargs,
    ):
        # Load vocab before calling super().__init__
        with open(vocab_file, "r") as f:
            data = json.load(f)

        # Support nested format with "token_to_id" key
        if isinstance(data, dict) and "token_to_id" in data:
            data = data["token_to_id"]

        # Ensure DNA bases are in the vocab (matching PlasmidTokenizer)
        next_id = max(data.values()) + 1 if data else 0
        for base in DNA_BASES:
            if base not in data:
                data[base] = next_id
                next_id += 1

        self._vocab = data
        self._id_to_token = {v: k for k, v in self._vocab.items()}

        # Only pass special tokens that actually exist in the vocab.
        # PreTrainedTokenizer would otherwise create new IDs for them.
        special_kwargs = {}
        for name, tok in [("bos_token", bos_token), ("eos_token", eos_token),
                          ("unk_token", unk_token), ("pad_token", pad_token),
                          ("sep_token", sep_token)]:
            if tok in self._vocab:
                special_kwargs[name] = tok

        # Disable automatic BOS/EOS insertion (transformers 5.x); prompts
        # already contain <BOS>...<SEP> and the model generates <EOS> itself.
        kwargs.setdefault("special_tokens_pattern", "none")
        super().__init__(**special_kwargs, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        """Split into special <...> tokens and individual characters."""
        parts = re.split(r"(<[^>]+>)", text)
        tokens = []
        for part in parts:
            if not part or part.isspace():
                continue
            if part.startswith("<") and part.endswith(">"):
                tokens.append(part)
            else:
                tokens.extend(c for c in part if not c.isspace())
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get("<UNK>", 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "<UNK>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

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
