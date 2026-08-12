#!/usr/bin/env python
"""Build reproducible FineWeb token slices for the Semantic-RQ LM experiment.

The manifest is independent of the model being evaluated.  For each context
n-gram ending at position ``t`` (the representation that predicts token
``t + 1``), it records one of:

* exact_seen: the same compressed n-gram occurred in LM training;
* semantic_neighbor: exact unseen, address covered, and its nearest covered
  training n-gram exceeds a frozen embedding-similarity threshold;
* covered_no_neighbor: address covered but no sufficiently similar source;
* address_oov: absent from the static RQ dictionary.

It also stores nearest-neighbor cosine, lexical overlap, and RQ-code overlap so
the model-loss analysis does not define semantic transfer circularly by codes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np


CATEGORY_NAMES = {
    0: "invalid",
    1: "exact_seen",
    2: "semantic_neighbor",
    3: "covered_no_neighbor",
    4: "address_oov",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--eval-rows", type=int, default=200)
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        help="Freeze streaming shuffle independently of eval_rows.",
    )
    parser.add_argument("--address-reserve-rows", type=int, default=6_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tokenize-batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--semantic-threshold-2", type=float, default=0.79)
    parser.add_argument("--semantic-threshold-3", type=float, default=0.76)
    parser.add_argument("--low-lexical-threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def poly_keys(compressed: np.ndarray, order: int, base: int) -> np.ndarray:
    """Return suffix polynomial keys aligned to the ending token position."""
    batch, length = compressed.shape
    result = np.zeros((batch, length), dtype=np.int64)
    if length < order:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(
        compressed, window_shape=order, axis=1
    )
    values = np.zeros(windows.shape[:2], dtype=np.int64)
    for offset in range(order):
        values = values * base + windows[..., offset].astype(np.int64)
    result[:, order - 1 :] = values
    return result


def table_positions(keys: np.ndarray, table_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(table_keys, keys)
    clipped = np.clip(positions, 0, max(0, len(table_keys) - 1))
    hit = (positions < len(table_keys)) & (table_keys[clipped] == keys)
    return clipped, hit


def valid_context_mask(attention: np.ndarray, order: int) -> np.ndarray:
    """Mask context endings that predict a real next token and contain no prefix pad."""
    mask = np.zeros_like(attention, dtype=bool)
    if attention.shape[1] <= order:
        return mask
    mask[:, order - 1 : -1] = (
        attention[:, order - 1 : -1].astype(bool)
        & attention[:, order:].astype(bool)
    )
    return mask


def initial_categories(
    valid: np.ndarray, covered: np.ndarray, exact: np.ndarray
) -> np.ndarray:
    """Assign mutually exclusive base slices; exact exposure has precedence."""
    category = np.zeros(valid.shape, dtype=np.uint8)
    category[valid & ~covered] = 4
    category[valid & covered & ~exact] = 3
    category[valid & exact] = 1
    return category


def trigram_jaccard(left: str, right: str) -> float:
    def grams(text: str) -> set[str]:
        normalized = " ".join(text.lower().split())
        if len(normalized) < 3:
            return {normalized}
        return {normalized[index : index + 3] for index in range(len(normalized) - 2)}

    a, b = grams(left), grams(right)
    return len(a & b) / max(1, len(a | b))


def embed_texts(
    model_name: str, texts: list[str], batch_size: int
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.float16
    ).cuda().eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            ).to("cuda")
            hidden = model(**batch).last_hidden_state
            last = batch["attention_mask"].sum(dim=1) - 1
            vectors = hidden[torch.arange(hidden.shape[0], device="cuda"), last]
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
            chunks.append(vectors.cpu().numpy())
            print(
                f"[slice embed] {min(start + batch_size, len(texts))}/{len(texts)}",
                flush=True,
            )
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks).astype(np.float32)


def batched_tokenize(tokenizer: Any, texts: list[str], batch_size: int, max_length: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np",
        )
        yield start, np.asarray(encoded["input_ids"], dtype=np.int64), np.asarray(
            encoded["attention_mask"], dtype=np.uint8
        )


def first_representatives(
    keys: np.ndarray,
    ids: np.ndarray,
    valid: np.ndarray,
    order: int,
) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    rows, columns = np.nonzero(valid)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        key = int(keys[row, column])
        if key not in result:
            result[key] = tuple(
                int(value) for value in ids[row, column - order + 1 : column + 1]
            )
    return result


def decode_representatives(tokenizer: Any, values: list[tuple[int, ...]]) -> list[str]:
    return list(tokenizer.batch_decode(values, skip_special_tokens=False))


def main() -> None:
    import faiss
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer

    args = parse_args()
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    orders = [int(value) for value in meta["ngram_sizes"]]
    table_keys = {
        order: np.load(args.table_dir / f"keys_{order}.npy", allow_pickle=False)
        for order in orders
    }
    table_codes = {
        order: np.load(args.table_dir / f"codes_{order}.npy", allow_pickle=False)
        for order in orders
    }

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1

    print(
        f"[slice data] FineWeb seed={args.seed}, reserve={args.address_reserve_rows}, "
        f"train={args.train_rows}, eval={args.eval_rows}",
        flush=True,
    )
    raw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    ).skip(args.address_reserve_rows).shuffle(
        seed=args.seed,
        buffer_size=(
            args.shuffle_buffer_size
            if args.shuffle_buffer_size is not None
            else max(args.train_rows + args.eval_rows, 20_000)
        ),
    )
    rows = [
        str(example["text"])
        for example in islice(raw, args.train_rows + args.eval_rows)
        if example.get("text")
    ]
    expected = args.train_rows + args.eval_rows
    if len(rows) != expected:
        raise RuntimeError(f"FineWeb yielded {len(rows)} usable rows, expected {expected}")
    train_texts, eval_texts = rows[: args.train_rows], rows[args.train_rows :]

    eval_ids_parts: list[np.ndarray] = []
    eval_attention_parts: list[np.ndarray] = []
    eval_keys_parts: dict[int, list[np.ndarray]] = {order: [] for order in orders}
    eval_valid_parts: dict[int, list[np.ndarray]] = {order: [] for order in orders}
    eval_representatives: dict[int, dict[int, tuple[int, ...]]] = {
        order: {} for order in orders
    }
    for _, ids, attention in batched_tokenize(
        tokenizer, eval_texts, args.tokenize_batch_size, args.max_length
    ):
        compressed = np.asarray(compressor.map_ids(ids), dtype=np.int64)
        eval_ids_parts.append(ids)
        eval_attention_parts.append(attention)
        for order in orders:
            keys = poly_keys(compressed, order, base)
            valid = valid_context_mask(attention, order)
            eval_keys_parts[order].append(keys)
            eval_valid_parts[order].append(valid)
            reps = first_representatives(keys, ids, valid, order)
            for key, value in reps.items():
                eval_representatives[order].setdefault(key, value)

    eval_ids = np.concatenate(eval_ids_parts)
    eval_attention = np.concatenate(eval_attention_parts)
    eval_keys = {order: np.concatenate(eval_keys_parts[order]) for order in orders}
    eval_valid = {order: np.concatenate(eval_valid_parts[order]) for order in orders}
    eval_key_sets = {
        order: set(map(int, np.unique(eval_keys[order][eval_valid[order]])))
        for order in orders
    }

    exact_seen: dict[int, set[int]] = {order: set() for order in orders}
    train_table_hit = {
        order: np.zeros(len(table_keys[order]), dtype=bool) for order in orders
    }
    train_table_access_count = {
        order: np.zeros(len(table_keys[order]), dtype=np.int64) for order in orders
    }
    train_representative_ids: dict[int, np.ndarray] = {
        order: np.full((len(table_keys[order]), order), -1, dtype=np.int64)
        for order in orders
    }
    for start, ids, attention in batched_tokenize(
        tokenizer, train_texts, args.tokenize_batch_size, args.max_length
    ):
        compressed = np.asarray(compressor.map_ids(ids), dtype=np.int64)
        for order in orders:
            keys = poly_keys(compressed, order, base)
            valid = valid_context_mask(attention, order)
            unique = np.unique(keys[valid])
            exact_seen[order].update(eval_key_sets[order].intersection(map(int, unique)))
            positions, covered = table_positions(keys, table_keys[order])
            covered &= valid
            rows_index, columns = np.nonzero(covered)
            for row, column in zip(rows_index.tolist(), columns.tolist(), strict=True):
                table_index = int(positions[row, column])
                if train_table_hit[order][table_index]:
                    continue
                train_table_hit[order][table_index] = True
                train_representative_ids[order][table_index] = ids[
                    row, column - order + 1 : column + 1
                ]
            np.add.at(
                train_table_access_count[order], positions[covered], 1
            )
        if start % (args.tokenize_batch_size * 100) == 0:
            print(f"[slice scan] {start}/{args.train_rows} train rows", flush=True)

    categories: dict[int, np.ndarray] = {}
    similarities: dict[int, np.ndarray] = {}
    lexical_scores: dict[int, np.ndarray] = {}
    code_overlaps: dict[int, np.ndarray] = {}
    shared_head_masks: dict[int, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    thresholds = {2: args.semantic_threshold_2, 3: args.semantic_threshold_3}

    for order in orders:
        keys = eval_keys[order]
        valid = eval_valid[order]
        similarity = np.full(keys.shape, np.nan, dtype=np.float32)
        lexical = np.full(keys.shape, np.nan, dtype=np.float32)
        code_overlap = np.full(keys.shape, np.nan, dtype=np.float32)
        shared_head_mask = np.zeros(keys.shape, dtype=np.uint16)
        positions, covered = table_positions(keys, table_keys[order])
        covered &= valid
        exact = valid & np.isin(keys, np.fromiter(exact_seen[order], dtype=np.int64))
        category = initial_categories(valid, covered, exact)

        unseen_covered_keys = sorted(
            set(map(int, keys[covered & ~exact]))
        )
        train_indices = np.flatnonzero(train_table_hit[order])
        if unseen_covered_keys and len(train_indices):
            train_ids = [tuple(map(int, row)) for row in train_representative_ids[order][train_indices]]
            train_surfaces = decode_representatives(tokenizer, train_ids)
            eval_rep_ids = [eval_representatives[order][key] for key in unseen_covered_keys]
            eval_surfaces = decode_representatives(tokenizer, eval_rep_ids)
            all_embeddings = embed_texts(
                args.embedder,
                train_surfaces + eval_surfaces,
                args.embed_batch_size,
            )
            train_embeddings = all_embeddings[: len(train_surfaces)]
            query_embeddings = all_embeddings[len(train_surfaces) :]
            index = faiss.IndexFlatIP(train_embeddings.shape[1])
            index.add(np.ascontiguousarray(train_embeddings))
            nearest_similarity, nearest_local = index.search(
                np.ascontiguousarray(query_embeddings), 1
            )
            nearest_similarity = nearest_similarity[:, 0]
            nearest_train_indices = train_indices[nearest_local[:, 0]]
            eval_table_positions, _ = table_positions(
                np.asarray(unseen_covered_keys, dtype=np.int64), table_keys[order]
            )
            nearest_head_matches = (
                table_codes[order][eval_table_positions]
                == table_codes[order][nearest_train_indices]
            )
            if nearest_head_matches.shape[1] > 16:
                raise ValueError("shared-head bitmask supports at most 16 RQ levels")
            nearest_code_overlap = np.mean(nearest_head_matches, axis=1)
            nearest_head_bits = np.zeros(len(nearest_head_matches), dtype=np.uint16)
            for level in range(nearest_head_matches.shape[1]):
                nearest_head_bits |= (
                    nearest_head_matches[:, level].astype(np.uint16) << level
                )
            nearest_lexical = np.asarray(
                [
                    trigram_jaccard(eval_surface, train_surfaces[int(local)])
                    for eval_surface, local in zip(
                        eval_surfaces, nearest_local[:, 0], strict=True
                    )
                ],
                dtype=np.float32,
            )
            key_to_values = {
                key: (float(sem), float(lex), float(overlap), int(head_bits))
                for key, sem, lex, overlap, head_bits in zip(
                    unseen_covered_keys,
                    nearest_similarity,
                    nearest_lexical,
                    nearest_code_overlap,
                    nearest_head_bits,
                    strict=True,
                )
            }
            for key, (sem, lex, overlap, head_bits) in key_to_values.items():
                mask = valid & (keys == key)
                similarity[mask] = sem
                lexical[mask] = lex
                code_overlap[mask] = overlap
                shared_head_mask[mask] = head_bits
                category[mask] = 2 if sem >= thresholds[order] else 3

        categories[order] = category
        similarities[order] = similarity
        lexical_scores[order] = lexical
        code_overlaps[order] = code_overlap
        shared_head_masks[order] = shared_head_mask
        counts = {
            name: int(np.sum(category == code))
            for code, name in CATEGORY_NAMES.items()
            if code
        }
        sem_mask = category == 2
        low_lex = sem_mask & (lexical <= args.low_lexical_threshold)
        summaries[str(order)] = {
            "valid_context_tokens": int(valid.sum()),
            "train_exact_eval_keys": len(exact_seen[order]),
            "train_covered_unique_keys": int(train_table_hit[order].sum()),
            "counts": counts,
            "semantic_neighbor_low_lexical": int(low_lex.sum()),
            "semantic_neighbor_mean_similarity": (
                float(np.nanmean(similarity[sem_mask])) if sem_mask.any() else None
            ),
            "semantic_neighbor_mean_code_overlap": (
                float(np.nanmean(code_overlap[sem_mask])) if sem_mask.any() else None
            ),
            "semantic_neighbor_shared_code": int(
                np.sum(sem_mask & (shared_head_mask > 0))
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "input_ids": eval_ids.astype(np.int32),
        "attention_mask": eval_attention.astype(np.uint8),
    }
    for order in orders:
        arrays[f"category_{order}"] = categories[order]
        arrays[f"semantic_similarity_{order}"] = similarities[order].astype(np.float16)
        arrays[f"lexical_jaccard_{order}"] = lexical_scores[order].astype(np.float16)
        arrays[f"rq_code_overlap_{order}"] = code_overlaps[order].astype(np.float16)
        arrays[f"rq_shared_head_mask_{order}"] = shared_head_masks[order]
        arrays[f"train_access_count_{order}"] = train_table_access_count[order]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)

    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "protocol": {
            "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
            "address_reserve_rows": args.address_reserve_rows,
            "train_rows": args.train_rows,
            "eval_rows": args.eval_rows,
            "shuffle_buffer_size": (
                args.shuffle_buffer_size
                if args.shuffle_buffer_size is not None
                else max(args.train_rows + args.eval_rows, 20_000)
            ),
            "max_length": args.max_length,
            "embedder": args.embedder,
            "semantic_thresholds": {str(key): value for key, value in thresholds.items()},
            "low_lexical_threshold": args.low_lexical_threshold,
            "causal_alignment": "context n-gram ending at t is assigned to loss predicting token t+1",
        },
        "orders": summaries,
        "manifest": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = args.summary.with_suffix(args.summary.suffix + ".tmp")
    summary_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(summary_tmp, args.summary)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
