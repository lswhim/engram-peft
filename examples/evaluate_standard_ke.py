#!/usr/bin/env python
"""Standard CounterFact/ZsRE batch-edit evaluation with full-target likelihood."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["counterfact", "zsre"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engram-weights")
    parser.add_argument("--lora-weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means the complete official split")
    return parser.parse_args()


def flatten_answer(value: Any) -> str:
    while isinstance(value, list) and value:
        value = value[0]
    return str(value) if value is not None else ""


def load_cases(dataset: str, limit: int) -> list[dict[str, Any]]:
    if dataset == "counterfact":
        source = load_dataset("azhx/counterfact", split="test")
        if limit:
            source = source.select(range(min(limit, len(source))))
        cases = []
        for index, example in enumerate(source):
            rewrite = example["requested_rewrite"]
            true = rewrite["target_true"]["str"]
            cases.append(
                {
                    "case_id": int(example.get("case_id", index)),
                    "prompt": rewrite["prompt"].format(rewrite["subject"]),
                    "target_new": rewrite["target_new"]["str"],
                    "target_true": true,
                    "paraphrases": list(example["paraphrase_prompts"]),
                    "neighbors": [(prompt, true) for prompt in example["neighborhood_prompts"]],
                }
            )
        return cases

    path = Path("data/zsre/benchmark/ZsRE/ZsRE-test-all.json")
    source = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        source = source[:limit]
    cases = []
    for index, example in enumerate(source):
        neighbors = []
        for neighbor in example.get("locality", {}).get("Relation_Specificity", []):
            answer = flatten_answer(neighbor.get("ground_truth"))
            if answer:
                neighbors.append((str(neighbor["prompt"]), answer))
        rephrase = example.get("rephrase_prompt")
        cases.append(
            {
                "case_id": index,
                "prompt": str(example["prompt"]),
                "target_new": flatten_answer(example["target_new"]),
                "target_true": flatten_answer(example.get("ground_truth")),
                "paraphrases": [str(rephrase)] if rephrase else [],
                "neighbors": neighbors,
            }
        )
    return cases


@torch.inference_mode()
def sequence_logprob(tokenizer: Any, model: Any, prompt: str, answer: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    if not answer_ids:
        return float("-inf")
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)
    logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1].float()
    start = len(prompt_ids) - 1
    positions = torch.arange(start, start + len(answer_ids), device=model.device)
    targets = input_ids[0, len(prompt_ids) :]
    return float(torch.log_softmax(logits[positions], dim=-1).gather(1, targets[:, None]).sum())


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def main() -> None:
    args = parse_args()
    if args.engram_weights and args.lora_weights:
        raise ValueError("load either Engram or LoRA, not both")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    model: Any = base
    if args.engram_weights:
        from engram_peft import EngramModel

        model = EngramModel.from_pretrained(
            base, args.engram_weights, tokenizer=tokenizer
        ).to("cuda")
    elif args.lora_weights:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.lora_weights).to("cuda")
    model.eval()
    cases = load_cases(args.dataset, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples_path = args.output.with_suffix(".jsonl")
    counts = {"efficacy": 0.0, "paraphrase": 0.0, "specificity": 0.0}
    denominators = {"efficacy": 0, "paraphrase": 0, "specificity": 0}
    with samples_path.open("w", encoding="utf-8") as handle:
        for case_index, case in enumerate(cases, 1):
            new_lp = sequence_logprob(tokenizer, model, case["prompt"], case["target_new"])
            true_lp = sequence_logprob(tokenizer, model, case["prompt"], case["target_true"])
            efficacy = float(new_lp > true_lp)
            counts["efficacy"] += efficacy
            denominators["efficacy"] += 1
            paraphrase_results = []
            for prompt in case["paraphrases"]:
                margin = sequence_logprob(tokenizer, model, prompt, case["target_new"]) - sequence_logprob(tokenizer, model, prompt, case["target_true"])
                paraphrase_results.append({"prompt": prompt, "margin": margin, "success": float(margin > 0)})
            neighbor_results = []
            for prompt, answer in case["neighbors"]:
                margin = sequence_logprob(tokenizer, model, prompt, answer) - sequence_logprob(tokenizer, model, prompt, case["target_new"])
                neighbor_results.append({"prompt": prompt, "margin": margin, "success": float(margin > 0)})
            for key, results in (("paraphrase", paraphrase_results), ("specificity", neighbor_results)):
                counts[key] += sum(item["success"] for item in results)
                denominators[key] += len(results)
            handle.write(json.dumps({
                "case_id": case["case_id"], "efficacy": efficacy,
                "efficacy_margin": new_lp - true_lp,
                "paraphrases": paraphrase_results, "neighbors": neighbor_results,
            }, ensure_ascii=False) + "\n")
            if case_index % 100 == 0:
                print(f"[{case_index}/{len(cases)}]", flush=True)
    metrics = {
        key: counts[key] / denominators[key] if denominators[key] else None
        for key in counts
    }
    score_values = [float(metrics[key]) for key in ("efficacy", "paraphrase", "specificity") if metrics[key] is not None]
    metrics["harmonic_score"] = harmonic(score_values)
    payload = {
        "status": "complete", "dataset": args.dataset, "examples": len(cases),
        "complete_official_split": args.limit == 0, "metrics": metrics,
        "denominators": denominators, "samples": str(samples_path),
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
