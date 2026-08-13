#!/usr/bin/env python
"""Encode missing n-grams with a frozen semantic encoder and persisted RQ codebooks.

The input is one or more JSONL files. Their text is tokenized exactly like Engram,
missing compressed 2/3-grams are embedded in batches, then encoded by the original
frozen FAISS residual quantizer. The output is a new complete immutable table; the
source table is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.build_rq_table import (
    build_compressor,
    embed_texts,
    poly_key,
    unpack_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-files", type=Path, nargs="+", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_table.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    meta = json.loads((source / "meta.json").read_text())
    for n in meta["ngram_sizes"]:
        if not (source / f"rq_{n}.faiss").is_file():
            raise FileNotFoundError(
                f"{source / f'rq_{n}.faiss'} is missing; rebuild the source table "
                "with the persisted frozen RQ codebook"
            )

    comp, tokenizer = build_compressor(meta["base_tokenizer"])
    base = int(meta["base"])
    tokenizer_pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    compressed_pad_id = int(comp.map_ids(np.asarray([tokenizer_pad_id]))[0])
    missing: dict[int, dict[int, list[int]]] = {
        int(n): {} for n in meta["ngram_sizes"]
    }
    existing = {
        int(n): set(np.load(source / f"keys_{n}.npy").tolist())
        for n in meta["ngram_sizes"]
    }
    for data_file in args.data_files:
        with data_file.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get(args.text_column)
                if not text:
                    continue
                ids = tokenizer(
                    text, truncation=True, max_length=args.max_doc_tokens
                )["input_ids"]
                cids = comp.map_ids(np.asarray(ids)).tolist()
                for n in missing:
                    pad = [compressed_pad_id] * (n - 1)
                    padded_cids = pad + cids
                    padded_ids = [tokenizer_pad_id] * (n - 1) + ids
                    for end in range(n - 1, len(padded_cids)):
                        cwindow = padded_cids[end - n + 1 : end + 1]
                        key = poly_key(cwindow, base)
                        if key not in existing[n] and key not in missing[n]:
                            missing[n][key] = padded_ids[end - n + 1 : end + 1]

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta.json", output / "meta.json")
    import faiss

    for n in missing:
        old_keys = np.load(source / f"keys_{n}.npy")
        old_codes = np.load(source / f"codes_{n}.npy")
        new_keys = np.asarray(sorted(missing[n]), dtype=np.int64)
        if len(new_keys):
            texts = [
                tokenizer.decode(missing[n][int(key)], skip_special_tokens=False)
                for key in new_keys
            ]
            args.embedder = meta["embedder"]
            embeddings = embed_texts(args, texts)
            index = faiss.read_index(str(source / f"rq_{n}.faiss"))
            packed = index.sa_encode(np.ascontiguousarray(embeddings.astype(np.float32)))
            nbits = int(np.log2(int(meta["codebook_size"])))
            new_codes = unpack_codes(packed, int(meta["num_levels"]), nbits).astype(
                old_codes.dtype
            )
            keys = np.concatenate([old_keys, new_keys])
            codes = np.concatenate([old_codes, new_codes])
            order = np.argsort(keys)
            keys, codes = keys[order], codes[order]
        else:
            keys, codes = old_keys, old_codes
        np.save(output / f"keys_{n}.npy", keys)
        np.save(output / f"codes_{n}.npy", codes)
        shutil.copy2(source / f"rq_{n}.faiss", output / f"rq_{n}.faiss")
        print(f"[{n}-gram] added={len(new_keys)} total={len(keys)}", flush=True)

    extended_meta = json.loads((output / "meta.json").read_text())
    extended_meta["strict_semantic_rq"] = True
    extended_meta["extended_from"] = str(source)
    extended_meta["coverage_files"] = [str(path.resolve()) for path in args.data_files]
    (output / "meta.json").write_text(json.dumps(extended_meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
