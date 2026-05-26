#!/usr/bin/env python
"""
Offline builder for the semantic-hash (RQ) Engram address table.

All steps are OFFLINE and the result is frozen, so runtime stays O(1) / no neural compute.

  1. Build CompressedTokenizer for the base model tokenizer (same normalization Engram
     uses at runtime).
  2. Stream a corpus, extract suffix n-grams over *compressed* token ids, count freq,
     keep top-N per size.
  3. Embed each kept n-gram's surface text with an external embedder (Qwen3-Embedding).
     This is the only semantic step; it happens here once.
  4. Train a faiss ResidualQuantizer (M levels x K codes) per n-gram size; encode each
     n-gram into M integer codes.
  5. Save {sorted int64 key -> M codes} arrays + meta. OOV at runtime falls back to the
     arithmetic hash, so the RQ object itself need not be persisted.

Key encoding (per n-gram size n): k = sum_j c_j * base^(n-1-j), base = V'+1, int64.
With V'~1e5 and n<=3 this stays < int64 max.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RQ semantic-hash table for Engram.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", default="text")
    p.add_argument("--num_docs", type=int, default=20000)
    p.add_argument("--max_doc_tokens", type=int, default=512)
    p.add_argument("--base_tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--ngram_sizes", type=int, nargs="+", default=[2, 3])
    p.add_argument("--num_levels", type=int, default=8,
                   help="RQ levels M per n-gram size (= per-ngram head count).")
    p.add_argument("--codebook_size", type=int, default=256, help="codes per level (K)")
    p.add_argument("--max_ngrams_per_size", type=int, default=500000)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--embed_batch_size", type=int, default=256)
    p.add_argument("--output_dir", default="rq_tables/fineweb_qwen3")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_compressor(base_tokenizer: str):
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer

    tok = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    comp = CompressedTokenizer(tokenizer=tok)
    print(f"[compressor] V={comp.vocab_size} -> V'={comp.compressed_vocab_size}")
    return comp, tok


def poly_key(comp_ids, base: int) -> int:
    k = 0
    for c in comp_ids:
        k = k * base + int(c)
    return k


def scan_ngrams(args, comp, tok):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    base = comp.compressed_vocab_size + 1
    counters = {n: Counter() for n in args.ngram_sizes}
    repr_tokens = {n: {} for n in args.ngram_sizes}
    seen = 0
    for ex in ds:
        text = ex.get(args.text_column)
        if not text:
            continue
        ids = tok(text, truncation=True, max_length=args.max_doc_tokens)["input_ids"]
        if len(ids) < max(args.ngram_sizes):
            continue
        cids = comp.map_ids(np.asarray(ids)).tolist()
        for n in args.ngram_sizes:
            for i in range(n - 1, len(cids)):
                key = poly_key(cids[i - n + 1 : i + 1], base)
                counters[n][key] += 1
                if key not in repr_tokens[n]:
                    repr_tokens[n][key] = ids[i - n + 1 : i + 1]
        seen += 1
        if seen >= args.num_docs:
            break
    print(f"[scan] {seen} docs; "
          + ", ".join(f"{n}-gram uniq={len(counters[n])}" for n in args.ngram_sizes))
    return base, counters, repr_tokens


def select_ngrams(args, counters, repr_tokens, tok):
    selected = {}
    for n in args.ngram_sizes:
        items = [(k, c) for k, c in counters[n].items() if c >= args.min_count]
        items.sort(key=lambda kc: kc[1], reverse=True)
        items = items[: args.max_ngrams_per_size]
        keys = np.array([k for k, _ in items], dtype=np.int64)
        keys.sort()
        texts = [tok.decode(repr_tokens[n][int(k)], skip_special_tokens=False)
                 for k in keys]
        selected[n] = {"keys": keys, "texts": texts}
        print(f"[select] {n}-gram kept {len(keys)}")
    return selected


def embed_texts(args, texts):
    import torch
    from transformers import AutoModel, AutoTokenizer

    etok = AutoTokenizer.from_pretrained(args.embedder, trust_remote_code=True)
    emodel = AutoModel.from_pretrained(
        args.embedder, trust_remote_code=True, dtype=torch.float16
    ).cuda().eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), args.embed_batch_size):
            enc = etok(texts[s : s + args.embed_batch_size], padding=True,
                       truncation=True, max_length=32, return_tensors="pt").to("cuda")
            hs = emodel(**enc).last_hidden_state
            lengths = enc["attention_mask"].sum(dim=1) - 1
            vec = hs[torch.arange(hs.size(0)), lengths]
            vec = torch.nn.functional.normalize(vec, dim=-1)
            out.append(vec.float().cpu().numpy())
            if (s // args.embed_batch_size) % 20 == 0:
                print(f"[embed] {s}/{len(texts)}")
    return np.concatenate(out, axis=0)


def unpack_codes(packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    N = packed.shape[0]
    bits = np.unpackbits(packed, axis=1, bitorder="big")
    codes = np.zeros((N, M), dtype=np.int64)
    for m in range(M):
        chunk = bits[:, m * nbits : (m + 1) * nbits]
        val = np.zeros(N, dtype=np.int64)
        for b in range(nbits):
            val = (val << 1) | chunk[:, b].astype(np.int64)
        codes[:, m] = val
    return codes


def train_rq(args, emb):
    import faiss

    d = emb.shape[1]
    nbits = int(np.log2(args.codebook_size))
    assert (1 << nbits) == args.codebook_size, "codebook_size must be power of 2"
    rq = faiss.ResidualQuantizer(d, args.num_levels, nbits)
    x = np.ascontiguousarray(emb.astype(np.float32))
    rq.train(x)
    packed = rq.compute_codes(x)
    return unpack_codes(packed, args.num_levels, nbits).astype(np.uint16)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    comp, tok = build_compressor(args.base_tokenizer)
    base, counters, repr_tokens = scan_ngrams(args, comp, tok)
    selected = select_ngrams(args, counters, repr_tokens, tok)

    meta = {
        "base": int(base),
        "compressed_vocab_size": int(comp.compressed_vocab_size),
        "ngram_sizes": args.ngram_sizes,
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "embedder": args.embedder,
        "base_tokenizer": args.base_tokenizer,
    }
    for n in args.ngram_sizes:
        keys, texts = selected[n]["keys"], selected[n]["texts"]
        if len(keys) == 0:
            print(f"[warn] no {n}-grams kept; skipping")
            continue
        emb = embed_texts(args, texts)
        codes = train_rq(args, emb)
        np.save(os.path.join(args.output_dir, f"keys_{n}.npy"), keys)
        np.save(os.path.join(args.output_dir, f"codes_{n}.npy"), codes)
        print(f"[save] {n}-gram keys{keys.shape} codes{codes.shape}")

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] -> {args.output_dir}")


if __name__ == "__main__":
    main()
