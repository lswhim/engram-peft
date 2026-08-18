#!/usr/bin/env python
"""Backfill semantic RQ residual-gain artifacts without changing frozen codes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from engram_peft.rq_table_tools import residual_energy_gains
from scripts.build_rq_table import (
    build_compressor,
    embed_texts,
    scan_ngrams,
    select_ngrams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--data-files", required=True)
    parser.add_argument("--num-docs", type=int, required=True)
    parser.add_argument("--max-doc-tokens", type=int, default=128)
    parser.add_argument("--max-ngrams-per-size", type=int, default=300000)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--text-columns", nargs="+", default=["prompt", "target"])
    parser.add_argument("--propagate-to", type=Path, nargs="*", default=[])
    return parser.parse_args()


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("projection", "none") != "none":
        raise NotImplementedError("backfill currently requires projection='none'")
    build_args = SimpleNamespace(
        base_tokenizer=meta["base_tokenizer"],
        embedder=meta["embedder"],
        ngram_sizes=[int(value) for value in meta["ngram_sizes"]],
        data_files=args.data_files,
        dataset=None,
        dataset_config=None,
        split="train",
        text_column="text",
        text_columns=args.text_columns,
        num_docs=args.num_docs,
        max_doc_tokens=args.max_doc_tokens,
        max_ngrams_per_size=args.max_ngrams_per_size,
        min_count=args.min_count,
        embed_batch_size=args.embed_batch_size,
        filter_language=None,
        filter_domain=None,
        min_edu_score=None,
    )
    compressor, tokenizer = build_compressor(build_args.base_tokenizer)
    base, counters, repr_tokens, documents_scanned = scan_ngrams(
        build_args, compressor, tokenizer
    )
    if int(base) != int(meta["base"]):
        raise RuntimeError(f"compressed-token base mismatch: {base} != {meta['base']}")
    selected = select_ngrams(build_args, counters, repr_tokens, tokenizer)

    import faiss

    for n in build_args.ngram_sizes:
        expected_keys = np.load(args.table_dir / f"keys_{n}.npy", allow_pickle=False)
        selected_keys = selected[n]["keys"]
        if not np.array_equal(selected_keys, expected_keys):
            raise RuntimeError(
                f"selected {n}-gram keys do not exactly reproduce the frozen table"
            )
        vectors = embed_texts(build_args, selected[n]["texts"])
        codes = np.load(args.table_dir / f"codes_{n}.npy", allow_pickle=False)
        index = faiss.read_index(str(args.table_dir / f"rq_{n}.faiss"))
        codebooks = faiss.vector_to_array(index.rq.codebooks).reshape(
            int(meta["num_levels"]), int(meta["codebook_size"]), vectors.shape[1]
        )
        gains = residual_energy_gains(vectors, codes, codebooks).astype(np.float16)
        atomic_save(args.table_dir / f"residual_gains_{n}.npy", gains)
        for destination in args.propagate_to:
            destination_keys = np.load(
                destination / f"keys_{n}.npy", allow_pickle=False
            )
            if not np.array_equal(destination_keys, expected_keys):
                raise RuntimeError(f"key mismatch in propagated table {destination}")
            atomic_save(destination / f"residual_gains_{n}.npy", gains)
        print(
            json.dumps(
                {
                    "ngram_size": n,
                    "rows": len(codes),
                    "documents_scanned": documents_scanned,
                    "mean_gain_by_level": gains.astype(np.float32).mean(axis=0).tolist(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
