#!/usr/bin/env python
"""Audit whether WikiBigEdit RQ addresses have a useful coarse-to-fine geometry.

This is a benchmark-level address audit, not a model-quality smoke test.  It scans the
full chronological manifest, dynamically encodes every previously unseen n-gram with
the frozen semantic encoder/RQ codebook, and compares semantic codes with the exact
runtime-shuffled control.  The output is a machine-readable JSON report plus a compact
HTML report suitable for the paper dashboard.

The decisive tests are:

* prefix collision load at every RQ depth;
* terminal and anywhere-in-prompt prefix overlap for canonical -> rephrase pairs;
* the same measurements for locality and deterministic random controls;
* relation-template and subject purity of terminal address buckets.

No model parameter is read or updated.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0, help="0 scans all cases")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, default=None)
    return parser.parse_args()


def stable_int(text: str, seed: int) -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8"), digest_size=8, person=f"wiki-rq:{seed}".encode()[:16]
    ).digest()
    return int.from_bytes(digest, "little")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def relation_signature(prompt: str, subject: str) -> str:
    normalized = normalize(prompt)
    subject_norm = normalize(subject)
    if subject_norm:
        normalized = normalized.replace(subject_norm, "<subject>")
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", normalized)
    return normalized


@dataclass
class Case:
    case_id: str
    prompt: str
    subject: str
    relation: str
    propagation: list[str]
    locality: list[str]


def load_cases(path: Path, max_cases: int) -> list[Case]:
    cases: list[Case] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            prompt = str(row["prompt"])
            subject = str(metadata.get("subject") or "")
            propagation: list[str] = []
            locality: list[str] = []
            for query in row.get("queries") or []:
                query_prompt = str(query.get("prompt") or "").strip()
                if not query_prompt:
                    continue
                role = query.get("role")
                if role == "should_propagate":
                    propagation.append(query_prompt)
                elif role == "should_not_propagate":
                    locality.append(query_prompt)
            cases.append(
                Case(
                    case_id=str(row.get("case_id", len(cases))),
                    prompt=prompt,
                    subject=subject,
                    relation=relation_signature(prompt, subject),
                    propagation=propagation,
                    locality=locality,
                )
            )
            if max_cases and len(cases) >= max_cases:
                break
    if not cases:
        raise ValueError(f"no cases loaded from {path}")
    return cases


def batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    compressor: Any,
    mapping: Any,
    batch_size: int,
    max_length: int,
) -> list[dict[int, np.ndarray]]:
    """Return one [positions, levels] code matrix per order for every text."""
    outputs: list[dict[int, np.ndarray]] = []
    completed = 0
    for chunk in batches(texts, batch_size):
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        original = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention = np.asarray(encoded["attention_mask"], dtype=np.int64)
        compressed = np.asarray(
            [compressor.map_ids(row) for row in original], dtype=np.int64
        )
        all_codes = mapping.hash(compressed, original_ids=original)[0]
        level_count = mapping.num_levels
        for row_index in range(len(chunk)):
            length = int(attention[row_index].sum())
            per_order: dict[int, np.ndarray] = {}
            for order_index, order in enumerate(mapping.ngram_sizes):
                start = order_index * level_count
                stop = start + level_count
                # Left padding is absent in the source tokenizer configuration used by
                # this project.  Retain only real-token positions.
                per_order[int(order)] = all_codes[row_index, :length, start:stop].copy()
            outputs.append(per_order)
        completed += len(chunk)
        print(f"[encode] {completed}/{len(texts)}", flush=True)
    return outputs


def shuffled_codes(codes: np.ndarray, order: int, levels: int, k: int, seed: int) -> np.ndarray:
    output = np.empty_like(codes, dtype=np.int64)
    flat = codes.reshape(-1, levels)
    shuffled = output.reshape(-1, levels)
    for index, row in enumerate(flat):
        digest = hashlib.blake2b(
            np.asarray(row, dtype=np.uint32).tobytes(),
            digest_size=8 * levels,
            person=f"rq{seed}:{order}".encode()[:16],
        ).digest()
        shuffled[index] = (
            np.frombuffer(digest, dtype=np.uint64) % k
        ).astype(np.int64)
    return output


def common_prefix(left: np.ndarray, right: np.ndarray) -> int:
    equal = left == right
    mismatch = np.flatnonzero(~equal)
    return int(mismatch[0]) if len(mismatch) else int(len(equal))


def pair_prefix_stats(left: np.ndarray, right: np.ndarray) -> tuple[int, int]:
    """Return terminal prefix and max prefix over any position pair."""
    terminal = common_prefix(left[-1], right[-1])
    best = 0
    # K is tiny (16) and prompts are capped at 128 tokens.  Vectorizing per depth
    # keeps the full-manifest analysis tractable without quadratic Python loops.
    max_depth = left.shape[1]
    for depth in range(1, max_depth + 1):
        left_prefix = {tuple(row[:depth]) for row in left}
        if any(tuple(row[:depth]) in left_prefix for row in right):
            best = depth
        else:
            break
    return terminal, best


def distribution(values: list[int], levels: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean_prefix_depth": float(array.mean()) if len(array) else None,
        "median_prefix_depth": float(np.median(array)) if len(array) else None,
        "p90_prefix_depth": float(np.quantile(array, 0.9)) if len(array) else None,
        "full_match_rate": float(np.mean(array == levels)) if len(array) else None,
        "at_least_depth": {
            str(depth): float(np.mean(array >= depth)) if len(array) else None
            for depth in range(1, levels + 1)
        },
    }


def weighted_bucket_purity(bucket_labels: dict[tuple[int, ...], Counter[str]]) -> float:
    numerator = sum(max(labels.values()) for labels in bucket_labels.values() if labels)
    denominator = sum(sum(labels.values()) for labels in bucket_labels.values())
    return numerator / denominator if denominator else float("nan")


def normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(len(counts))


def terminal_bucket_report(
    canonical: list[dict[int, np.ndarray]],
    cases: list[Case],
    order: int,
    levels: int,
    *,
    shuffled_seed: int | None = None,
    codebook_size: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for depth in range(1, levels + 1):
        relation_buckets: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
        subject_buckets: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
        loads: Counter[tuple[int, ...]] = Counter()
        for case, code_map in zip(cases, canonical, strict=True):
            codes = code_map[order]
            if shuffled_seed is not None:
                if codebook_size is None:
                    raise ValueError("codebook_size is required for shuffled buckets")
                codes = shuffled_codes(
                    codes, order, levels, codebook_size, shuffled_seed
                )
            key = tuple(map(int, codes[-1, :depth]))
            loads[key] += 1
            relation_buckets[key][case.relation] += 1
            subject_buckets[key][normalize(case.subject)] += 1
        load_array = np.asarray(list(loads.values()), dtype=np.float64)
        relation_entropy = np.average(
            [normalized_entropy(labels) for labels in relation_buckets.values()],
            weights=[sum(labels.values()) for labels in relation_buckets.values()],
        )
        subject_entropy = np.average(
            [normalized_entropy(labels) for labels in subject_buckets.values()],
            weights=[sum(labels.values()) for labels in subject_buckets.values()],
        )
        report[str(depth)] = {
            "unique_prefixes": len(loads),
            "collision_rate": 1.0 - len(loads) / len(cases),
            "mean_bucket_load": float(load_array.mean()),
            "p95_bucket_load": float(np.quantile(load_array, 0.95)),
            "max_bucket_load": int(load_array.max()),
            "relation_purity": weighted_bucket_purity(relation_buckets),
            "subject_purity": weighted_bucket_purity(subject_buckets),
            "relation_normalized_entropy": float(relation_entropy),
            "subject_normalized_entropy": float(subject_entropy),
        }
    return report


def table_collision_report(table_dir: Path, order: int, levels: int) -> dict[str, Any]:
    codes = np.load(table_dir / f"codes_{order}.npy", allow_pickle=False).astype(np.int64)
    report: dict[str, Any] = {}
    for depth in range(1, levels + 1):
        _, counts = np.unique(codes[:, :depth], axis=0, return_counts=True)
        report[str(depth)] = {
            "rows": int(len(codes)),
            "unique_prefixes": int(len(counts)),
            "collision_rate": float(1.0 - len(counts) / len(codes)),
            "mean_bucket_load": float(counts.mean()),
            "p95_bucket_load": float(np.quantile(counts, 0.95)),
            "max_bucket_load": int(counts.max()),
        }
    return report


def compute_pair_report(
    cases: list[Case],
    canonical: list[dict[int, np.ndarray]],
    encoded_by_text: dict[str, dict[int, np.ndarray]],
    orders: list[int],
    levels: int,
    shuffled_seed: int | None,
    codebook_size: int,
    random_seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for order in orders:
        role_values: dict[str, dict[str, list[int]]] = {
            role: {"terminal": [], "anywhere": []}
            for role in ("propagation", "locality", "random")
        }
        shuffled_values: dict[str, dict[str, list[int]]] = {
            role: {"terminal": [], "anywhere": []}
            for role in ("propagation", "locality", "random")
        }
        for case_index, case in enumerate(cases):
            source = canonical[case_index][order]
            random_index = stable_int(case.case_id, random_seed) % len(cases)
            if random_index == case_index:
                random_index = (random_index + 1) % len(cases)
            targets = {
                "propagation": case.propagation,
                "locality": case.locality,
                "random": [cases[random_index].prompt],
            }
            if shuffled_seed is not None:
                shuffled_source = shuffled_codes(
                    source, order, levels, codebook_size, shuffled_seed
                )
            for role, queries in targets.items():
                for query in queries:
                    target = encoded_by_text[query][order]
                    terminal, anywhere = pair_prefix_stats(source, target)
                    role_values[role]["terminal"].append(terminal)
                    role_values[role]["anywhere"].append(anywhere)
                    if shuffled_seed is not None:
                        shuffled_target = shuffled_codes(
                            target, order, levels, codebook_size, shuffled_seed
                        )
                        s_terminal, s_anywhere = pair_prefix_stats(
                            shuffled_source, shuffled_target
                        )
                        shuffled_values[role]["terminal"].append(s_terminal)
                        shuffled_values[role]["anywhere"].append(s_anywhere)
        output[str(order)] = {
            "semantic": {
                role: {
                    position: distribution(values, levels)
                    for position, values in positions.items()
                }
                for role, positions in role_values.items()
            },
            "shuffled": {
                role: {
                    position: distribution(values, levels)
                    for position, values in positions.items()
                }
                for role, positions in shuffled_values.items()
            }
            if shuffled_seed is not None
            else None,
        }
    return output


def render_html(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    levels = payload["protocol"]["num_levels"]
    for order, order_data in payload["pair_overlap"].items():
        for method in ("semantic", "shuffled"):
            if not order_data.get(method):
                continue
            for role in ("propagation", "locality", "random"):
                terminal = order_data[method][role]["terminal"]
                anywhere = order_data[method][role]["anywhere"]
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(order)}-gram</td><td>{html.escape(method)}</td>"
                    f"<td>{html.escape(role)}</td>"
                    f"<td>{terminal['mean_prefix_depth']:.3f}</td>"
                    f"<td>{anywhere['mean_prefix_depth']:.3f}</td>"
                    f"<td>{100 * anywhere['full_match_rate']:.2f}%</td>"
                    "</tr>"
                )
    purity_rows: list[str] = []
    for order, methods in payload["terminal_buckets"].items():
        for method, depths in methods.items():
            for depth, stats in depths.items():
                purity_rows.append(
                    "<tr>"
                    f"<td>{html.escape(order)}-gram</td><td>{html.escape(method)}</td>"
                    f"<td>{depth}/{levels}</td>"
                    f"<td>{stats['unique_prefixes']:,}</td>"
                    f"<td>{100 * stats['collision_rate']:.2f}%</td>"
                    f"<td>{stats['relation_purity']:.3f}</td>"
                    f"<td>{stats['subject_purity']:.3f}</td>"
                    f"<td>{stats['p95_bucket_load']:.1f}</td>"
                    "</tr>"
                )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>WikiBigEdit RQ Hierarchy Audit</title>
<style>
body{{font:14px/1.5 system-ui;background:#f5f7fb;color:#172033;margin:0}}
main{{max-width:1180px;margin:auto;padding:32px}}section{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;padding:20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px 11px;border-bottom:1px solid #e8ecf2;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.muted{{color:#637083}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>WikiBigEdit · RQ Hierarchy Audit</h1>
<p class="muted">完整 manifest；无模型训练；生成于 {html.escape(payload['updated_at'])}</p>
<section><h2>协议</h2><p>Cases: {payload['protocol']['cases']:,} · RQ: M={levels}, K={payload['protocol']['codebook_size']} · dynamic cache enabled</p></section>
<section><h2>Canonical → Query prefix overlap</h2>
<table><thead><tr><th>Order</th><th>Address</th><th>Pair role</th><th>Terminal mean depth</th><th>Anywhere mean depth</th><th>Anywhere full match</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Canonical terminal buckets</h2>
<table><thead><tr><th>Order</th><th>Address</th><th>Prefix</th><th>Unique</th><th>Collision</th><th>Relation purity</th><th>Subject purity</th><th>P95 load</th></tr></thead><tbody>{''.join(purity_rows)}</tbody></table></section>
<section><h2>判读原则</h2><p>支持层级假设需要：随着 prefix 加深，collision/load 平滑下降；relation purity 在浅层高于 random/shuffled structure；propagation overlap 明显高于 locality/random；Semantic 与 Shuffled 的差异不能只来自相同 n-gram 的确定性复用。</p></section>
</main></body></html>"""


def main() -> None:
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer
    from engram_peft.rq_hashing import RQNgramMapping

    args = parse_args()
    cases = load_cases(args.manifest, args.max_cases)
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    shuffled_meta = json.loads(
        (args.shuffled_dir / "meta.json").read_text(encoding="utf-8")
    )
    shuffled_seed = shuffled_meta.get("runtime_shuffle_seed")
    if shuffled_seed is None:
        raise ValueError("shuffled table metadata lacks runtime_shuffle_seed")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    mapping = RQNgramMapping(
        table_dir=str(args.table_dir),
        cache_dir=str(args.cache_dir),
        embed_batch_size=args.batch_size,
    )

    all_query_texts = sorted(
        {
            query
            for case in cases
            for query in [*case.propagation, *case.locality]
        }
    )
    canonical_texts = [case.prompt for case in cases]
    unique_texts = list(dict.fromkeys([*canonical_texts, *all_query_texts]))
    encoded = encode_texts(
        unique_texts,
        tokenizer,
        compressor,
        mapping,
        args.batch_size,
        args.max_length,
    )
    encoded_by_text = dict(zip(unique_texts, encoded, strict=True))
    canonical = [encoded_by_text[text] for text in canonical_texts]
    orders = [int(order) for order in mapping.ngram_sizes]

    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "manifest": str(args.manifest),
            "table_dir": str(args.table_dir),
            "shuffled_dir": str(args.shuffled_dir),
            "cases": len(cases),
            "unique_texts": len(unique_texts),
            "num_levels": mapping.num_levels,
            "codebook_size": mapping.codebook_size,
            "ngram_sizes": orders,
            "max_length": args.max_length,
            "seed": args.seed,
            "random_control": "deterministic different-case canonical prompt",
            "terminal": "last real-token suffix address",
            "anywhere": "maximum shared ordered RQ prefix over all source/query positions",
        },
        "case_coverage": {
            "with_propagation": sum(bool(case.propagation) for case in cases),
            "with_locality": sum(bool(case.locality) for case in cases),
            "propagation_queries": sum(len(case.propagation) for case in cases),
            "locality_queries": sum(len(case.locality) for case in cases),
        },
        "table_collisions": {
            str(order): table_collision_report(
                args.table_dir, order, mapping.num_levels
            )
            for order in orders
        },
        "terminal_buckets": {
            str(order): {
                "semantic": terminal_bucket_report(
                    canonical, cases, order, mapping.num_levels
                ),
                "shuffled": terminal_bucket_report(
                    canonical,
                    cases,
                    order,
                    mapping.num_levels,
                    shuffled_seed=int(shuffled_seed),
                    codebook_size=mapping.codebook_size,
                ),
            }
            for order in orders
        },
        "pair_overlap": compute_pair_report(
            cases,
            canonical,
            encoded_by_text,
            orders,
            mapping.num_levels,
            int(shuffled_seed),
            mapping.codebook_size,
            args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    html_output = args.html_output or args.output.with_suffix(".html")
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "html": str(html_output)}), flush=True)


if __name__ == "__main__":
    main()
