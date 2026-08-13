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
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--case-chunk-size", type=int, default=32)
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
def sequence_logprobs(
    tokenizer: Any, model: Any, pairs: list[tuple[str, str]], batch_size: int
) -> list[float]:
    """Score complete answer strings in batches, masking every prompt token."""
    encoded: list[tuple[list[int], int]] = []
    for prompt, answer in pairs:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
        encoded.append((prompt_ids + answer_ids, len(prompt_ids)))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    scores: list[float] = []
    for offset in range(0, len(encoded), batch_size):
        batch = encoded[offset : offset + batch_size]
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=model.device)
        attention_mask = torch.zeros_like(input_ids)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=model.device)
            attention_mask[row, : len(ids)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1].float()
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:, None]).squeeze(-1)
        for row, (ids, prompt_len) in enumerate(batch):
            if len(ids) <= prompt_len:
                scores.append(float("-inf"))
            else:
                scores.append(float(token_logprobs[row, prompt_len - 1 : len(ids) - 1].sum()))
    return scores


def evaluate_chunk(tokenizer: Any, model: Any, cases: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    pairs: list[tuple[str, str]] = []
    specs: list[tuple[int, str]] = []
    for case_index, case in enumerate(cases):
        comparisons = [("efficacy", case["prompt"], case["target_new"], case["target_true"])]
        comparisons += [("paraphrase", prompt, case["target_new"], case["target_true"]) for prompt in case["paraphrases"]]
        comparisons += [("specificity", prompt, answer, case["target_new"]) for prompt, answer in case["neighbors"]]
        for kind, prompt, positive, negative in comparisons:
            specs.extend([(case_index, kind), (case_index, kind)])
            pairs.extend([(prompt, positive), (prompt, negative)])
    scores = sequence_logprobs(tokenizer, model, pairs, batch_size)
    results = [{"efficacy": [], "paraphrase": [], "specificity": []} for _ in cases]
    for offset in range(0, len(specs), 2):
        case_index, kind = specs[offset]
        results[case_index][kind].append(scores[offset] - scores[offset + 1])
    return results


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
        for chunk_start in range(0, len(cases), args.case_chunk_size):
            chunk = cases[chunk_start : chunk_start + args.case_chunk_size]
            evaluated = evaluate_chunk(tokenizer, model, chunk, args.batch_size)
            for case, result in zip(chunk, evaluated):
                efficacy_margin = result["efficacy"][0]
                efficacy = float(efficacy_margin > 0)
                counts["efficacy"] += efficacy
                denominators["efficacy"] += 1
                paraphrase_results = [
                    {"prompt": prompt, "margin": margin, "success": float(margin > 0)}
                    for prompt, margin in zip(case["paraphrases"], result["paraphrase"])
                ]
                neighbor_results = [
                    {"prompt": prompt, "margin": margin, "success": float(margin > 0)}
                    for (prompt, _), margin in zip(case["neighbors"], result["specificity"])
                ]
                for key, items in (("paraphrase", paraphrase_results), ("specificity", neighbor_results)):
                    counts[key] += sum(item["success"] for item in items)
                    denominators[key] += len(items)
                handle.write(json.dumps({
                    "case_id": case["case_id"], "efficacy": efficacy,
                    "efficacy_margin": efficacy_margin,
                    "paraphrases": paraphrase_results, "neighbors": neighbor_results,
                }, ensure_ascii=False) + "\n")
            complete = min(chunk_start + len(chunk), len(cases))
            print(f"[{complete}/{len(cases)}]", flush=True)
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
        "scoring": "full_target_conditional_log_likelihood",
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
