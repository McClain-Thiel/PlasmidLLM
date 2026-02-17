"""HuggingFace-compatible tokenizer for PlasmidLM."""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from transformers import PreTrainedTokenizer


class PlasmidLMTokenizer(PreTrainedTokenizer):
    """Simple word-level tokenizer for plasmid sequences."""

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
            self._vocab = json.load(f)
        self._id_to_token = {v: k for k, v in self._vocab.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        """Split on whitespace and angle-bracket tokens."""
        tokens = re.findall(r"<[^>]+>|[^\s<>]+", text)
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
