#!/usr/bin/env python
"""
MMLU evaluation via lm-evaluation-harness for base or Engram-wrapped models.

  # base model only (feasibility check)
  python examples/eval_mmlu.py --tasks mmlu_anatomy --limit 20

  # engram checkpoint (rebuilds engram from saved config incl. hash_backend/rq_table_dir)
  python examples/eval_mmlu.py --engram_weights outputs/benchmarks/engram_weights \
      --tasks mmlu_anatomy --limit 20
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--engram_weights", default=None,
                    help="Dir with a saved Engram checkpoint; omit for base model.")
    ap.add_argument("--tasks", default="mmlu_anatomy",
                    help="Comma-separated lm-eval task names (e.g. 'mmlu' for full).")
    ap.add_argument("--limit", type=int, default=20,
                    help="Max examples per task.")
    ap.add_argument("--batch_size", default="8")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda"
    )

    model = base
    tag = "base"
    if args.engram_weights:
        from engram_peft import EngramModel

        model = EngramModel.from_pretrained(base, args.engram_weights, tokenizer=tok)
        model.eval()
        tag = f"engram:{args.engram_weights}"
        print(f"[eval_mmlu] loaded Engram checkpoint from {args.engram_weights}")

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    res = simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit)

    print("===EVAL_RESULTS===", tag)
    for task, metrics in res["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc", "n/a"))
        print(f"RESULT  {task}  acc={acc}")
    print("===EVAL_DONE===")


if __name__ == "__main__":
    main()
