#!/usr/bin/env python3
"""Classifier-free-guidance decoding for PlasmidLM, to test whether F16 is fixable at
inference time.

F16 diagnosis: the model reverts to the component that co-occurs with the backbone in
training (40% of the time) instead of the requested substitute, even though the substitute
is well-supported. That is a conditioning-strength failure — the prompt is being overruled
by the co-occurrence prior — not a knowledge failure. CFG attacks exactly that.

At each DNA-generation step, run two streams sharing the generated suffix:
    conditional    <BOS> <requested component tokens> <SEQ> <generated DNA...>
    unconditional  <BOS> <SEQ> <generated DNA...>
and decode from the guided logits
    guided = uncond + w * (cond - uncond)
w=1 is ordinary conditional sampling (the current model). w>1 amplifies the prompt's pull
away from the unconditional prior. If raising w lifts substituted-in-component recovery,
the failure is steerable without retraining.

Both streams are one padded batch; we sample ONE token per prompt from the guided logits and
feed it to both rows, so the two caches stay in lock-step on an identical suffix.

Not a change to the core model — a standalone re-implementation of the KV-cached decode loop
that reads the model's public inner forward.

Usage:
  python cfg_generate.py --model <path-or-hf> --prompts novel_benchmark.parquet \\
      --guidance 1.0 2.0 3.0 --n 64 --out-dir cfg_run --max-new-tokens 2500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_dna(text: str) -> str:
    return "".join(c for c in text.upper() if c in "ACGT")


def encode(tokenizer, prompts, device):
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.inference_mode()
def cfg_generate(model, cond_ids, cond_mask, uncond_ids, uncond_mask, *,
                 w, max_new_tokens, temperature, top_k, top_p, eos_id, pad_id):
    """Guided decode. cond_* and uncond_* are (N, S_c) and (N, S_u), each left-padded."""
    N = cond_ids.shape[0]
    device = cond_ids.device

    def prefill(ids, mask):
        hidden, kv, _ = model.model(ids, None, 0, attention_mask=mask)
        return model.lm_head(hidden[:, -1:, :]).squeeze(1), kv, ids.shape[1]

    cond_logits, cond_kv, cond_len = prefill(cond_ids, cond_mask)
    unc_logits, unc_kv, unc_len = prefill(uncond_ids, uncond_mask)

    finished = torch.zeros(N, dtype=torch.bool, device=device)
    gen = []
    for _ in range(max_new_tokens):
        guided = unc_logits + w * (cond_logits - unc_logits)
        scaled = guided / max(temperature, 1e-7)
        if top_k > 0:
            k = min(top_k, scaled.size(-1))
            tv, _ = torch.topk(scaled, k)
            scaled[scaled < tv[:, [-1]]] = float("-inf")
        if top_p > 0.0:
            sl, si = torch.sort(scaled, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            mask = cp - F.softmax(sl, dim=-1) >= top_p
            sl[mask] = float("-inf")
            scaled = sl.scatter(1, si, sl)
        tok = torch.multinomial(F.softmax(scaled, dim=-1), 1)
        tok = torch.where(finished.unsqueeze(1), pad_id, tok)
        gen.append(tok)

        if eos_id is not None:
            finished = finished | (tok.squeeze(1) == eos_id)
            if finished.all():
                break

        ch, cond_kv, _ = model.model(tok, cond_kv, cond_len)
        cond_logits = model.lm_head(ch).squeeze(1)
        cond_len += 1
        uh, unc_kv, _ = model.model(tok, unc_kv, unc_len)
        unc_logits = model.lm_head(uh).squeeze(1)
        unc_len += 1

    return torch.cat(gen, dim=1) if gen else torch.empty(N, 0, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True).to(args.device).eval()

    prompts = pd.read_parquet(args.prompts, columns=["prompt"])["prompt"].tolist()[:args.n]
    # unconditional prompt: strip every component token, keep <BOS>...<SEQ> scaffold
    def strip_components(p: str) -> str:
        head = p.split("<SEQ>")[0]
        bos = "<BOS>" if head.startswith("<BOS>") else ""
        return bos + "<SEQ>"
    unconds = [strip_components(p) for p in prompts]

    print(f"model: {args.model}")
    print(f"prompts: {len(prompts)}   example uncond: {unconds[0]!r}")
    print(f"eos={tok.eos_token_id} pad={tok.pad_token_id}\n")

    for w in args.guidance:
        rows, t0 = [], time.time()
        for s in range(0, len(prompts), args.batch_size):
            cp = prompts[s:s + args.batch_size]
            up = unconds[s:s + args.batch_size]
            ci, cm = encode(tok, cp, args.device)
            ui, um = encode(tok, up, args.device)
            out = cfg_generate(
                model, ci, cm, ui, um, w=w,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id or 0)
            for j, prompt in enumerate(cp):
                raw = tok.decode(out[j].tolist())
                dna = clean_dna(raw.split("<SEQ>")[-1] if "<SEQ>" in raw else raw)
                rows.append({"prompt": prompt, "sequence": dna, "length": len(dna),
                             "gc": (dna.count("G") + dna.count("C")) / max(len(dna), 1),
                             "guidance": w})
            print(f"  w={w}  [{len(rows)}/{len(prompts)}]  "
                  f"{len(rows)/(time.time()-t0)*60:.1f} seq/min", flush=True)

        out_dir = args.out_dir / f"w{w}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "generations.parquet", index=False)
        with open(out_dir / "generations.fasta", "w") as fh:
            for i, sq in enumerate(df.sequence):
                if sq:
                    fh.write(f">gen_{i}\n{sq}\n")
        print(f"  w={w}: median {df.length.median():.0f}bp -> {out_dir}\n")


if __name__ == "__main__":
    main()
