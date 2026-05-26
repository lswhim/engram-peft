#!/bin/bash
# Build RQ table for Qwen3-0.6B tokenizer (shares Qwen3 tokenizer w/ 1.7B, but build
# fresh so the poly-key base = V'+1 matches at runtime).
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
cd /anguszhang-cfs-nj/seokliu_workspace/engram
CUDA_VISIBLE_DEVICES=7 $PY scripts/build_rq_table.py \
  --dataset almanach/Biomed-Enriched --dataset_config none --split commercial \
  --filter_language en --filter_domain biomedical --min_edu_score 4.0 \
  --num_docs 5000 --max_ngrams_per_size 100000 --min_count 2 \
  --base_tokenizer Qwen/Qwen3-0.6B --embedder Qwen/Qwen3-Embedding-0.6B \
  --num_levels 8 --codebook_size 256 --output_dir rq_tables/biomed_qwen3_06b
echo DONE_BUILD06B_EXIT_$?
