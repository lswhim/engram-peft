#!/bin/bash
# Qwen3-1.7B matrix: {arithmetic, rq} x {seed 2024, 2025} on GPU 0-3.
# Aligned w/ TinyEngram: target_layers=[5,7,13,17], Biomed edu>4.0.
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
RQ=/anguszhang-cfs-nj/seokliu_workspace/engram/rq_tables/biomed_qwen3
M=Qwen/Qwen3-1.7B
LOG=/anguszhang-cfs-nj/seokliu_workspace
cd /anguszhang-cfs-nj/seokliu_workspace/engram
COMMON="--dataset biomed --model_name $M --max_steps 3000 --subset 30000 --batch_size 16"
ARITH="engram:n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[5,7,13,17]"
RQM="engram:hash_backend=rq,rq_table_dir=$RQ,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[5,7,13,17]"

CUDA_VISIBLE_DEVICES=0 nohup $PY examples/compare_engram_lora.py $COMMON --seed 2024 --methods $ARITH > $LOG/t_17b_arith_2024.log 2>&1 &
echo PID_17b_arith_2024 $!
CUDA_VISIBLE_DEVICES=1 nohup $PY examples/compare_engram_lora.py $COMMON --seed 2024 --methods $RQM   > $LOG/t_17b_rq_2024.log 2>&1 &
echo PID_17b_rq_2024 $!
CUDA_VISIBLE_DEVICES=2 nohup $PY examples/compare_engram_lora.py $COMMON --seed 2025 --methods $ARITH > $LOG/t_17b_arith_2025.log 2>&1 &
echo PID_17b_arith_2025 $!
CUDA_VISIBLE_DEVICES=3 nohup $PY examples/compare_engram_lora.py $COMMON --seed 2025 --methods $RQM   > $LOG/t_17b_rq_2025.log 2>&1 &
echo PID_17b_rq_2025 $!
