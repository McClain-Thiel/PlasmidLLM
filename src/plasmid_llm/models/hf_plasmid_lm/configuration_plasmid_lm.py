"""HuggingFace configuration for PlasmidLM."""

from transformers import PretrainedConfig


class PlasmidLMConfig(PretrainedConfig):
    model_type = "plasmid_lm"

    def __init__(
        self,
        vocab_size: int = 112,
        hidden_size: int = 384,
        num_hidden_layers: int = 10,
        num_attention_heads: int = 8,
        intermediate_size: int = 1536,
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 16384,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        super().__init__(
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
