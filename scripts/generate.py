"""Generate plasmid sequences using vLLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate plasmid sequences with vLLM")
    parser.add_argument("--hf-model", required=True, help="Path to HF model directory")
    parser.add_argument("--vocab", required=True, help="Path to vocab JSON")
    parser.add_argument(
        "--prompt",
        default="<BOS> <SEP>",
        help="Tag prompt (e.g. '<BOS> <COPY_HIGH> <AMR_KANAMYCIN> <ORI_COLE1> <SEP>')",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tokenizer = PlasmidLMTokenizer(args.vocab)
    prompt_ids = tokenizer.encode(args.prompt)

    print(f"Loading vLLM model: {args.hf_model}")
    llm = LLM(model=args.hf_model, trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    # Submit N copies of the prompt — vLLM batches them automatically
    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.num_samples} sequence(s)...\n")

    outputs = llm.generate(
        prompt_token_ids=[prompt_ids] * args.num_samples,
        sampling_params=sampling_params,
    )

    for i, output in enumerate(outputs):
        generated_ids = output.outputs[0].token_ids
        seq = tokenizer.decode(list(generated_ids))
        print(f"--- Sample {i + 1} ---")
        print(seq)
        print()


if __name__ == "__main__":
    main()
