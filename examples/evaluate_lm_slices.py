#!/usr/bin/env python
"""Evaluate a base/Engram checkpoint on a frozen LM token-slice manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


CATEGORIES = {
    1: "exact_seen",
    2: "semantic_neighbor",
    3: "covered_no_neighbor",
    4: "address_oov",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engram-weights", type=Path)
    parser.add_argument(
        "--result-suffix",
        help="Resolve engram_weights from a completed outputs/benchmarks result.",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--head-mask",
        choices=("none", "shared", "random-matched"),
        default="none",
        help=(
            "Semantic-RQ intervention: mask heads shared with the nearest train "
            "semantic neighbor, or an equal-count random set."
        ),
    )
    parser.add_argument("--low-lexical-threshold", type=float, default=0.10)
    parser.add_argument("--high-lexical-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_from_result(suffix: str) -> Path:
    for path in Path("outputs/benchmarks").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        params = payload.get("params")
        metrics = payload.get("metrics")
        if not isinstance(params, dict) or params.get("run_suffix") != suffix:
            continue
        save_dir = metrics.get("save_dir") if isinstance(metrics, dict) else None
        if isinstance(save_dir, str) and Path(save_dir).is_dir():
            return Path(save_dir)
    raise FileNotFoundError(f"no completed checkpoint for run_suffix={suffix!r}")


def metric(losses: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    selected = losses[mask]
    if not len(selected):
        return {"tokens": 0, "nll": None, "ppl": None}
    nll = float(selected.mean(dtype=np.float64))
    return {"tokens": int(len(selected)), "nll": nll, "ppl": float(math.exp(min(nll, 20.0)))}


def intervention_mask(
    data: Any,
    mode: str,
    num_heads_per_order: int,
    seed: int,
) -> np.ndarray | None:
    if mode == "none":
        return None
    orders = (2, 3)
    full = np.zeros(
        (
            len(data["input_ids"]),
            data["input_ids"].shape[1],
            len(orders) * num_heads_per_order,
        ),
        dtype=bool,
    )
    rng = np.random.default_rng(seed)
    for order_index, order in enumerate(orders):
        categories = np.asarray(data[f"category_{order}"])
        bits = np.asarray(data[f"rq_shared_head_mask_{order}"], dtype=np.uint16)
        bits = np.where(categories == 2, bits, 0)
        shared = np.stack(
            [
                (bits & np.uint16(1 << level)) != 0
                for level in range(num_heads_per_order)
            ],
            axis=-1,
        )
        selected = shared
        if mode == "random-matched":
            selected = np.zeros_like(shared)
            counts = shared.sum(axis=-1)
            scores = rng.random(shared.shape)
            for count in range(1, num_heads_per_order + 1):
                locations = np.nonzero(counts == count)
                if not len(locations[0]):
                    continue
                choices = np.argpartition(
                    scores[locations], kth=count - 1, axis=1
                )[:, :count]
                selected[locations[0][:, None], locations[1][:, None], choices] = True
        start = order_index * num_heads_per_order
        full[..., start : start + num_heads_per_order] = selected
    return full


def main() -> None:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.engram_weights is not None and args.result_suffix:
        raise ValueError("use only one of --engram-weights and --result-suffix")
    if args.result_suffix:
        args.engram_weights = checkpoint_from_result(args.result_suffix)
    data = np.load(args.manifest, allow_pickle=False)
    input_ids = np.asarray(data["input_ids"], dtype=np.int64)
    attention = np.asarray(data["attention_mask"], dtype=bool)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda().eval()
    model: Any = base
    if args.engram_weights:
        from engram_peft import EngramModel

        model = EngramModel.from_pretrained(
            base, str(args.engram_weights), tokenizer=tokenizer
        ).cuda().eval()
        print(f"[slice eval] loaded {args.engram_weights}", flush=True)

    head_mask = None
    mask_state: dict[str, Any] = {"value": None}
    hook_handles: list[Any] = []
    if args.head_mask != "none":
        if not args.engram_weights:
            raise ValueError("head masking requires --engram-weights/--result-suffix")
        if getattr(model.config, "hash_backend", None) != "rq":
            raise ValueError("shared-head intervention is defined only for Semantic-RQ")
        num_heads = int(model.config.n_head_per_ngram)
        head_mask = intervention_mask(data, args.head_mask, num_heads, args.seed)

        def mask_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            current = mask_state["value"]
            if current is None:
                raise RuntimeError("head-mask hook called without an active batch mask")
            if tuple(current.shape) != tuple(output.shape[:-1]):
                raise RuntimeError(
                    f"head-mask shape {tuple(current.shape)} != embedding shape {tuple(output.shape)}"
                )
            return output.masked_fill(current.unsqueeze(-1), 0)

        for layer in model.engram_layers.values():
            hook_handles.append(
                layer.multi_head_embedding.register_forward_hook(mask_hook)
            )
        print(
            f"[slice eval] intervention={args.head_mask}, "
            f"masked_head_activations={int(head_mask.sum())}",
            flush=True,
        )

    token_losses = np.full((len(input_ids), input_ids.shape[1] - 1), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(input_ids), args.batch_size):
            stop = min(start + args.batch_size, len(input_ids))
            ids = torch.from_numpy(input_ids[start:stop]).cuda()
            mask = torch.from_numpy(attention[start:stop]).cuda()
            if head_mask is not None:
                mask_state["value"] = torch.from_numpy(head_mask[start:stop]).cuda()
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            mask_state["value"] = None
            targets = ids[:, 1:]
            losses = functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape)
            token_losses[start:stop] = losses.cpu().numpy()
            print(f"[slice eval] {stop}/{len(input_ids)}", flush=True)

    for handle in hook_handles:
        handle.remove()

    valid_loss = attention[:, :-1] & attention[:, 1:]
    metrics: dict[str, Any] = {"overall": metric(token_losses, valid_loss)}
    for order in (2, 3):
        category = np.asarray(data[f"category_{order}"])[:, :-1]
        lexical = np.asarray(data[f"lexical_jaccard_{order}"], dtype=np.float32)[:, :-1]
        similarity = np.asarray(data[f"semantic_similarity_{order}"], dtype=np.float32)[:, :-1]
        overlap = np.asarray(data[f"rq_code_overlap_{order}"], dtype=np.float32)[:, :-1]
        per_category = {
            name: metric(token_losses, valid_loss & (category == code))
            for code, name in CATEGORIES.items()
        }
        low_lex_mask = valid_loss & (category == 2) & (lexical <= args.low_lexical_threshold)
        per_category["semantic_neighbor_low_lexical"] = metric(token_losses, low_lex_mask)
        sem_mask = valid_loss & (category == 2)
        shared_mask = sem_mask & (overlap > 0)
        no_shared_mask = sem_mask & (overlap == 0)
        per_category["semantic_neighbor_shared_code"] = metric(
            token_losses, shared_mask
        )
        per_category["semantic_neighbor_no_shared_code"] = metric(
            token_losses, no_shared_mask
        )
        per_category["semantic_neighbor_low_lexical_shared_code"] = metric(
            token_losses, low_lex_mask & (overlap > 0)
        )
        per_category["covered_no_neighbor_high_lexical"] = metric(
            token_losses,
            valid_loss
            & (category == 3)
            & (lexical >= args.high_lexical_threshold),
        )
        per_category["semantic_neighbor_diagnostics"] = {
            "tokens": int(sem_mask.sum()),
            "mean_similarity": float(np.nanmean(similarity[sem_mask])) if sem_mask.any() else None,
            "mean_rq_code_overlap": float(np.nanmean(overlap[sem_mask])) if sem_mask.any() else None,
        }
        metrics[str(order)] = per_category

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_path = args.output.with_suffix(".losses.npz")
    loss_tmp = loss_path.with_suffix(loss_path.suffix + ".tmp")
    with loss_tmp.open("wb") as handle:
        np.savez_compressed(handle, token_loss=token_losses)
    os.replace(loss_tmp, loss_path)
    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "engram_weights": str(args.engram_weights) if args.engram_weights else None,
        "manifest": str(args.manifest),
        "head_mask": args.head_mask,
        "masked_head_activations": int(head_mask.sum()) if head_mask is not None else 0,
        "losses": str(loss_path),
        "metrics": metrics,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
