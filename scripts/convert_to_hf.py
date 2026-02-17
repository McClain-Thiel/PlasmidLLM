"""Convert a PlasmidLLM checkpoint to HuggingFace format.

Creates a HF-compatible model directory with config.json, model.safetensors,
and tokenizer files that can be loaded with AutoModelForCausalLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.tokenizer import PlasmidTokenizer
from plasmid_llm.models import build_model
from omegaconf import OmegaConf


def convert_transformer_to_hf(checkpoint_path: str, vocab_path: str, output_dir: str):
    """Convert a transformer checkpoint to HuggingFace-compatible format."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    supported = ["transformer", "transformer_conv"]
    assert cfg.model.arch in supported, f"Expected one of {supported}, got {cfg.model.arch}"

    # Load tokenizer and model to get state dict keys right
    tokenizer = PlasmidTokenizer(vocab_path)
    model = build_model(cfg.model, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])

    # Map our state dict to HF-style naming
    # Keys use "model." prefix matching the standard HF convention:
    #   PlasmidLMForCausalLM.model = PlasmidLMModel (backbone)
    state_dict = model.state_dict()
    hf_state = {}

    # Embedding
    hf_state["model.embed_tokens.weight"] = state_dict["tok_emb.weight"]

    # Layers
    for key, val in state_dict.items():
        if key.startswith("blocks."):
            parts = key.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            if rest == "ln1.weight":
                hf_state[f"model.layers.{layer_idx}.input_layernorm.weight"] = val
            elif rest == "ln2.weight":
                hf_state[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = val
            elif rest == "attn.qkv.weight":
                # Split QKV into separate Q, K, V
                d_model = cfg.model.d_model
                q, k, v = val.split(d_model, dim=0)
                hf_state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q
                hf_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k
                hf_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
            elif rest == "attn.out_proj.weight":
                hf_state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = val
            elif rest == "mlp.0.weight":
                hf_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = val
            elif rest == "mlp.2.weight":
                hf_state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = val
            # Handle transformer_conv specific layers
            elif rest == "conv1d.weight":
                hf_state[f"model.layers.{layer_idx}.conv1d.weight"] = val
            elif rest == "conv1d.bias":
                hf_state[f"model.layers.{layer_idx}.conv1d.bias"] = val
            elif rest == "ln_conv.weight":
                hf_state[f"model.layers.{layer_idx}.ln_conv.weight"] = val

    # Final norm
    hf_state["model.norm.weight"] = state_dict["ln_f.weight"]

    # LM head omitted — tie_word_embeddings=True means HF (and vLLM) will
    # reuse embed_tokens. RoPE buffers are non-persistent (computed in __init__).

    # Save as safetensors
    print(f"Saving {len(hf_state)} tensors to {output / 'model.safetensors'}")
    save_file(hf_state, output / "model.safetensors")

    # Write config.json
    padded_vocab = ((tokenizer.vocab_size + 7) // 8) * 8
    hf_config = {
        "architectures": ["PlasmidLMForCausalLM"],
        "model_type": "plasmid_lm",
        "vocab_size": padded_vocab,
        "hidden_size": cfg.model.d_model,
        "num_hidden_layers": cfg.model.n_layers,
        "num_attention_heads": cfg.model.n_heads,
        "intermediate_size": cfg.model.d_ff,
        "hidden_act": "gelu",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 16384,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": tokenizer.vocab.get("<BOS>", 1),
        "eos_token_id": tokenizer.vocab.get("<EOS>", 2),
        "pad_token_id": tokenizer.vocab.get("<PAD>", 0),
        # Auto-map so AutoModelForCausalLM and AutoModel work with trust_remote_code
        "auto_map": {
            "AutoConfig": "configuration_plasmid_lm.PlasmidLMConfig",
            "AutoModel": "modeling_plasmid_lm.PlasmidLMModel",
            "AutoModelForCausalLM": "modeling_plasmid_lm.PlasmidLMForCausalLM",
            "AutoTokenizer": ["tokenization_plasmid_lm.PlasmidLMTokenizer", None],
        },
        # Custom fields for our architecture
        "_plasmid_llm": {
            "original_vocab_size": tokenizer.vocab_size,
            "checkpoint_step": ckpt.get("global_step", 0),
            "best_val_loss": ckpt.get("best_val_loss", None),
        },
    }
    with open(output / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Saved config.json")

    # Copy HF model code into the output directory
    hf_model_dir = Path(__file__).resolve().parent / "hf_model"
    for py_file in ["configuration_plasmid_lm.py", "modeling_plasmid_lm.py", "tokenization_plasmid_lm.py"]:
        src = hf_model_dir / py_file
        if src.exists():
            shutil.copy2(src, output / py_file)
    print(f"Copied model code for trust_remote_code")

    # Write tokenizer files
    vocab = tokenizer.vocab
    with open(output / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # Tokenizer config
    tokenizer_config = {
        "tokenizer_class": "PlasmidLMTokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization_plasmid_lm.PlasmidLMTokenizer", None],
        },
        "model_type": "plasmid_lm",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "sep_token": "<SEP>",
    }
    with open(output / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Generation config
    gen_config = {
        "max_new_tokens": 8192,
        "temperature": 0.8,
        "top_k": 50,
        "do_sample": True,
        "bos_token_id": hf_config["bos_token_id"],
        "eos_token_id": hf_config["eos_token_id"],
        "pad_token_id": hf_config["pad_token_id"],
    }
    with open(output / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    n_params = sum(v.numel() for v in hf_state.values())
    print(f"\nDone! HF model saved to {output}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Step: {ckpt.get('global_step', '?')}")
    print(f"  Best val loss: {ckpt.get('best_val_loss', '?')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_transformer_to_hf(args.checkpoint, args.vocab, args.output)


if __name__ == "__main__":
    main()
