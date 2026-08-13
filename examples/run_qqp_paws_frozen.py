"""Frozen Base/Engram representation benchmark: GLUE QQP -> PAWS.

The language model and, when present, the Stage-1 Engram memory are frozen.  Only
one shared-form binary linear head is trained on GLUE QQP.  Evaluation uses GLUE
QQP validation, PAWS-Wiki test, and optionally the officially reconstructed
PAWS-QQP ``dev_and_test.tsv`` (the upstream index archive is not redistributed).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--method",
        choices=["base", "arithmetic_matched", "rq_shuffled", "semantic_rq"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-suffix")
    parser.add_argument("--engram-weights", type=Path)
    parser.add_argument("--paws-qqp-tsv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    return parser.parse_args()


def completed_checkpoint(suffix: str) -> Path:
    candidates: list[Path] = []
    for filename in glob.glob("outputs/benchmarks/*.json"):
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics", {})
        if payload.get("params", {}).get("run_suffix") == suffix and (
            metrics.get("fixed_steps_complete") is True
            and metrics.get("completed_steps") == metrics.get("planned_steps") == 12_208
        ):
            checkpoint = metrics.get("save_dir")
            if isinstance(checkpoint, str) and Path(checkpoint).is_dir():
                candidates.append(Path(checkpoint))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one completed checkpoint for suffix={suffix!r}, got {candidates}"
        )
    return candidates[0]


def load_paws_qqp(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "question1": row.get("sentence1", ""),
                    "question2": row.get("sentence2", ""),
                    "label": int(row["label"]),
                }
            )
    if len(rows) != 677:
        raise ValueError(f"official PAWS-QQP dev_and_test must have 677 rows, got {len(rows)}")
    return Dataset.from_list(rows)


def result_is_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete" and payload.get("paper_eligible") is True


def normalize_rows(dataset: Dataset, source: str) -> Dataset:
    if source == "paws":
        return dataset.rename_columns({"sentence1": "question1", "sentence2": "question2"})
    return dataset


class PairCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [
            "Paraphrase classification.\n"
            f"Question 1: {row['question1']}\nQuestion 2: {row['question2']}"
            for row in rows
        ]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(row["label"]) for row in rows])
        return encoded


def pooled_hidden(model: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    lengths = batch["attention_mask"].sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    labels_t = torch.tensor(labels, dtype=torch.long)
    probs_t = torch.tensor(probabilities, dtype=torch.float64)
    predictions = (probs_t >= 0.5).long()
    tp = int(((predictions == 1) & (labels_t == 1)).sum())
    fp = int(((predictions == 1) & (labels_t == 0)).sum())
    fn = int(((predictions == 0) & (labels_t == 1)).sum())
    accuracy = float((predictions == labels_t).double().mean())
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    positive_count = int((labels_t == 1).sum())
    negative_count = len(labels) - positive_count
    order = torch.argsort(probs_t, stable=True)
    sorted_probs = probs_t[order]
    ranks = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    # Give tied scores their average rank without allocating an O(P*N) matrix.
    _, inverse, counts = torch.unique_consecutive(
        sorted_probs, return_inverse=True, return_counts=True
    )
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    average_ranks = offsets.double() + (counts.double() + 1.0) / 2.0
    ranks = average_ranks[inverse]
    positive_rank_sum = float(ranks[labels_t[order] == 1].sum())
    auroc = (
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
        if positive_count and negative_count
        else 0.0
    )
    return {"accuracy": accuracy, "f1": f1, "auroc": auroc, "examples": len(labels)}


@torch.inference_mode()
def evaluate(
    model: Any, head: nn.Module, loader: Iterable[Mapping[str, torch.Tensor]], device: torch.device
) -> dict[str, float]:
    model.eval()
    head.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    for cpu_batch in loader:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        hidden = pooled_hidden(model, batch)
        probabilities.extend(head(hidden.float()).softmax(dim=-1)[:, 1].cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return binary_metrics(labels, probabilities)


def main() -> None:
    args = parse_args()
    if args.result_suffix and args.engram_weights:
        raise ValueError("use only one of --result-suffix and --engram-weights")
    if args.method == "base" and (args.result_suffix or args.engram_weights):
        raise ValueError("base must not load Engram weights")
    if args.method != "base" and not (args.result_suffix or args.engram_weights):
        raise ValueError("Engram methods require a completed Stage-1 checkpoint")
    if args.result_suffix:
        args.engram_weights = completed_checkpoint(args.result_suffix)
    set_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model: Any = base
    if args.engram_weights:
        from engram_peft import EngramModel

        model = EngramModel.from_pretrained(base, str(args.engram_weights), tokenizer=tokenizer)
    model.requires_grad_(False)
    model.to(device).eval()
    hidden_size = int(base.config.hidden_size)
    head = nn.Linear(hidden_size, 2).to(device)

    qqp_train = load_dataset("nyu-mll/glue", "qqp", split="train")
    qqp_validation = load_dataset("nyu-mll/glue", "qqp", split="validation")
    paws_wiki = normalize_rows(
        load_dataset("google-research-datasets/paws", "labeled_final", split="test"),
        "paws",
    )
    if args.train_limit:
        qqp_train = qqp_train.select(range(min(args.train_limit, len(qqp_train))))
    if args.eval_limit:
        qqp_validation = qqp_validation.select(range(min(args.eval_limit, len(qqp_validation))))
        paws_wiki = paws_wiki.select(range(min(args.eval_limit, len(paws_wiki))))
    collator = PairCollator(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        qqp_train, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=True,
    )
    optimizer = AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_payload(payload: Mapping[str, Any]) -> None:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
        os.replace(temporary, args.output)

    status_payload: dict[str, Any] = {
        "status": "training",
        "protocol": "frozen Stage-1 representation; linear head trained on GLUE QQP only",
        "method": args.method,
        "seed": args.seed,
        "engram_weights": str(args.engram_weights) if args.engram_weights else None,
        "train_examples": len(qqp_train),
        "epochs": args.epochs,
        "total_steps": len(train_loader) * args.epochs,
        "completed_steps": 0,
        "paper_eligible": args.train_limit is None and args.eval_limit is None,
    }
    write_payload(status_payload)
    model.eval()
    for epoch in range(args.epochs):
        head.train()
        for step, cpu_batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with torch.no_grad():
                hidden = pooled_hidden(model, batch)
            loss = F.cross_entropy(head(hidden.float()), batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step = epoch * len(train_loader) + step
            if step % 100 == 0 or step == len(train_loader):
                status_payload.update(
                    completed_steps=global_step,
                    latest_loss=float(loss.item()),
                    wall_time_seconds=time.time() - started,
                )
                write_payload(status_payload)
            if step % 500 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.5f}", flush=True)

    def make_loader(dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator,
            num_workers=args.num_workers, pin_memory=True,
        )

    metrics: dict[str, Any] = {
        "qqp_validation": evaluate(model, head, make_loader(qqp_validation), device),
        "paws_wiki_test": evaluate(model, head, make_loader(paws_wiki), device),
    }
    if args.paws_qqp_tsv:
        metrics["paws_qqp_dev_and_test"] = evaluate(
            model, head, make_loader(load_paws_qqp(args.paws_qqp_tsv)), device
        )
    payload = {
        "status": "complete",
        "protocol": "frozen Stage-1 representation; linear head trained on GLUE QQP only",
        "method": args.method,
        "seed": args.seed,
        "engram_weights": str(args.engram_weights) if args.engram_weights else None,
        "train_examples": len(qqp_train),
        "epochs": args.epochs,
        "paper_eligible": args.train_limit is None and args.eval_limit is None,
        "metrics": metrics,
        "wall_time_seconds": time.time() - started,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    write_payload(payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
