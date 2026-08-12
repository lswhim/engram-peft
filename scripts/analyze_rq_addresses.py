#!/usr/bin/env python
"""Measure whether an RQ table organizes collisions by semantic similarity.

The analysis reconstructs the exact surface forms from the address-building
corpus, embeds a reproducible n-gram sample, compares Semantic-RQ with the
frequency-identical RQ-Shuffled control, and measures held-out address coverage.
It does not update any model parameter.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--sample-per-order", type=int, default=4000)
    parser.add_argument("--address-docs", type=int, default=5000)
    parser.add_argument("--coverage-skip", type=int, default=6000)
    parser.add_argument("--coverage-docs", type=int, default=200)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def poly_key(ids: Iterable[int], base: int) -> int:
    value = 0
    for token_id in ids:
        value = value * base + int(token_id)
    return value


def load_stream(skip: int = 0) -> Any:
    from datasets import load_dataset

    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    return stream.skip(skip) if skip else stream


def reconstruct_texts(
    args: argparse.Namespace,
    tokenizer: Any,
    compressor: Any,
    keys_by_order: dict[int, np.ndarray],
) -> dict[int, list[str]]:
    base = int(compressor.compressed_vocab_size) + 1
    targets = {order: set(map(int, keys)) for order, keys in keys_by_order.items()}
    found: dict[int, dict[int, str]] = {order: {} for order in targets}
    seen = 0
    for example in load_stream():
        text = example.get("text")
        if not text:
            continue
        ids = tokenizer(
            text, truncation=True, max_length=args.max_doc_tokens
        )["input_ids"]
        if len(ids) < max(targets):
            continue
        compressed = np.asarray(compressor.map_ids(np.asarray(ids)), dtype=np.int64)
        for order in targets:
            missing = targets[order] - found[order].keys()
            if not missing:
                continue
            for end in range(order - 1, len(compressed)):
                key = poly_key(compressed[end - order + 1 : end + 1], base)
                if key in missing:
                    found[order][key] = tokenizer.decode(
                        ids[end - order + 1 : end + 1], skip_special_tokens=False
                    )
        seen += 1
        if seen >= args.address_docs or all(
            len(found[order]) == len(targets[order]) for order in targets
        ):
            break
    result: dict[int, list[str]] = {}
    for order, keys in keys_by_order.items():
        missing = [int(key) for key in keys if int(key) not in found[order]]
        if missing:
            raise RuntimeError(
                f"could not reconstruct {len(missing)}/{len(keys)} sampled {order}-grams"
            )
        result[order] = [found[order][int(key)] for key in keys]
    return result


def reconstruct_texts_from_keys(
    tokenizer: Any,
    compressor: Any,
    keys_by_order: dict[int, np.ndarray],
) -> dict[int, list[str]]:
    """Decode a canonical representative when the source stream is unavailable."""
    reverse = np.full(int(compressor.compressed_vocab_size), -1, dtype=np.int64)
    for original_id, compressed_id in compressor.mapping.items():
        if reverse[int(compressed_id)] < 0:
            reverse[int(compressed_id)] = int(original_id)
    if np.any(reverse < 0):
        raise RuntimeError("compressed-token inverse is incomplete")
    base = int(compressor.compressed_vocab_size) + 1
    result: dict[int, list[str]] = {}
    for order, keys in keys_by_order.items():
        texts = []
        for raw_key in keys:
            key = int(raw_key)
            compressed_ids = [0] * order
            for position in range(order - 1, -1, -1):
                compressed_ids[position] = key % base
                key //= base
            original_ids = [int(reverse[value]) for value in compressed_ids]
            texts.append(tokenizer.decode(original_ids, skip_special_tokens=False))
        result[order] = texts
    return result


def embed_texts(args: argparse.Namespace, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.embedder, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.embedder, trust_remote_code=True, dtype=torch.float16
    ).cuda().eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), args.embed_batch_size):
            batch = tokenizer(
                texts[start : start + args.embed_batch_size],
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
            print(f"[diagnostic embed] {min(start + args.embed_batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(chunks).astype(np.float32)


def lexical_neighbors(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1).fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    neighbors.fit(matrix)
    return neighbors.kneighbors(matrix, return_distance=False)[:, 1]


def trigram_jaccard(left: str, right: str) -> float:
    def grams(text: str) -> set[str]:
        normalized = " ".join(text.lower().split())
        if len(normalized) < 3:
            return {normalized}
        return {normalized[index : index + 3] for index in range(len(normalized) - 2)}

    a, b = grams(left), grams(right)
    return len(a & b) / max(1, len(a | b))


def pair_diagnostics(
    texts: list[str],
    embeddings: np.ndarray,
    codes: np.ndarray,
    shuffled_codes: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    import faiss
    from scipy.stats import spearmanr

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    _, semantic_indices = index.search(np.ascontiguousarray(embeddings), 2)
    lexical_indices = lexical_neighbors(texts)
    random_indices = np.random.default_rng(seed).permutation(len(texts))

    pairs: set[tuple[int, int]] = set()
    for left in range(len(texts)):
        for right in (
            int(semantic_indices[left, 1]),
            int(lexical_indices[left]),
            int(random_indices[left]),
        ):
            if left != right:
                pairs.add((min(left, right), max(left, right)))

    ordered = np.asarray(sorted(pairs), dtype=np.int64)
    left, right = ordered[:, 0], ordered[:, 1]
    semantic = np.sum(embeddings[left] * embeddings[right], axis=1)
    lexical = np.asarray(
        [trigram_jaccard(texts[int(a)], texts[int(b)]) for a, b in ordered],
        dtype=np.float32,
    )
    rq_overlap = np.mean(codes[left] == codes[right], axis=1)
    shuffled_overlap = np.mean(shuffled_codes[left] == shuffled_codes[right], axis=1)
    semantic_low, semantic_high = np.quantile(semantic, [0.25, 0.75])
    lexical_low, lexical_high = np.quantile(lexical, [0.25, 0.75])
    masks = {
        "high_semantic_low_lexical": (semantic >= semantic_high) & (lexical <= lexical_low),
        "high_semantic_high_lexical": (semantic >= semantic_high) & (lexical >= lexical_high),
        "low_semantic_high_lexical": (semantic <= semantic_low) & (lexical >= lexical_high),
        "low_semantic_low_lexical": (semantic <= semantic_low) & (lexical <= lexical_low),
    }
    quadrants = {}
    for name, mask in masks.items():
        quadrants[name] = {
            "pairs": int(mask.sum()),
            "mean_semantic_cosine": float(semantic[mask].mean()) if mask.any() else None,
            "mean_lexical_jaccard": float(lexical[mask].mean()) if mask.any() else None,
            "rq_code_overlap": float(rq_overlap[mask].mean()) if mask.any() else None,
            "shuffled_code_overlap": float(shuffled_overlap[mask].mean()) if mask.any() else None,
        }
    rq_corr = spearmanr(semantic, rq_overlap).statistic
    shuffled_corr = spearmanr(semantic, shuffled_overlap).statistic
    return {
        "sampled_ngrams": len(texts),
        "candidate_pairs": len(ordered),
        "semantic_thresholds_q25_q75": [float(semantic_low), float(semantic_high)],
        "lexical_thresholds_q25_q75": [float(lexical_low), float(lexical_high)],
        "spearman_semantic_vs_rq_overlap": float(rq_corr),
        "spearman_semantic_vs_shuffled_overlap": float(shuffled_corr),
        "quadrants": quadrants,
    }


def address_coverage(
    args: argparse.Namespace,
    tokenizer: Any,
    compressor: Any,
    table_keys: dict[int, np.ndarray],
) -> dict[str, Any]:
    base = int(compressor.compressed_vocab_size) + 1
    totals = {order: 0 for order in table_keys}
    hits = {order: 0 for order in table_keys}
    seen = 0
    try:
        stream = load_stream(args.coverage_skip)
        for example in stream:
            text = example.get("text")
            if not text:
                continue
            ids = tokenizer(text, truncation=True, max_length=args.max_doc_tokens)["input_ids"]
            compressed = np.asarray(compressor.map_ids(np.asarray(ids)), dtype=np.int64)
            for order, keys in table_keys.items():
                if len(compressed) < order:
                    continue
                values = np.asarray(
                    [poly_key(compressed[end - order + 1 : end + 1], base) for end in range(order - 1, len(compressed))],
                    dtype=np.int64,
                )
                positions = np.searchsorted(keys, values)
                valid = positions < len(keys)
                matched = np.zeros(len(values), dtype=bool)
                matched[valid] = keys[positions[valid]] == values[valid]
                totals[order] += len(values)
                hits[order] += int(matched.sum())
            seen += 1
            if seen >= args.coverage_docs:
                break
    except Exception as error:
        return {
            "status": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
            "heldout_docs": 0,
            "skip_rows": args.coverage_skip,
            "per_order": {},
        }
    return {
        "status": "complete",
        "heldout_docs": seen,
        "skip_rows": args.coverage_skip,
        "per_order": {
            str(order): {
                "positions": totals[order],
                "hits": hits[order],
                "coverage": hits[order] / max(1, totals[order]),
            }
            for order in table_keys
        },
    }


def main() -> None:
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    orders = [int(value) for value in meta["ngram_sizes"]]
    table_keys: dict[int, np.ndarray] = {}
    sampled_keys: dict[int, np.ndarray] = {}
    sampled_codes: dict[int, np.ndarray] = {}
    sampled_shuffled: dict[int, np.ndarray] = {}
    for order in orders:
        keys = np.load(args.table_dir / f"keys_{order}.npy", allow_pickle=False)
        codes = np.load(args.table_dir / f"codes_{order}.npy", allow_pickle=False)
        shuffled = np.load(args.shuffled_dir / f"codes_{order}.npy", allow_pickle=False)
        count = min(args.sample_per_order, len(keys))
        indices = np.sort(rng.choice(len(keys), size=count, replace=False))
        table_keys[order] = keys
        sampled_keys[order] = keys[indices]
        sampled_codes[order] = codes[indices]
        sampled_shuffled[order] = shuffled[indices]

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    reconstruction = "observed_surface_from_address_corpus"
    try:
        texts = reconstruct_texts(args, tokenizer, compressor, sampled_keys)
    except Exception as error:
        reconstruction = "canonical_surface_from_compressed_key"
        print(
            f"[diagnostic] source stream unavailable ({type(error).__name__}: {error}); "
            "using canonical compressed-token representatives",
            flush=True,
        )
        texts = reconstruct_texts_from_keys(tokenizer, compressor, sampled_keys)
    all_texts = [text for order in orders for text in texts[order]]
    all_embeddings = embed_texts(args, all_texts)

    diagnostics: dict[str, Any] = {}
    offset = 0
    for order in orders:
        count = len(texts[order])
        diagnostics[str(order)] = pair_diagnostics(
            texts[order],
            all_embeddings[offset : offset + count],
            sampled_codes[order],
            sampled_shuffled[order],
            args.seed + order,
        )
        offset += count

    coverage = address_coverage(args, tokenizer, compressor, table_keys)
    payload = {
        "status": "complete" if coverage.get("status") == "complete" else "partial",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "sample_per_order": args.sample_per_order,
            "address_docs": args.address_docs,
            "pair_sources": ["semantic_nn", "lexical_nn", "random"],
            "quadrants": "within-order q25/q75 semantic cosine and char-trigram Jaccard",
            "embedder": args.embedder,
            "seed": args.seed,
            "surface_reconstruction": reconstruction,
        },
        "orders": diagnostics,
        "coverage": coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
