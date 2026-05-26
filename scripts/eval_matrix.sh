#!/bin/bash
# 3-way medical-MMLU eval: base / arithmetic-engram / RQ-engram
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
cd /anguszhang-cfs-nj/seokliu_workspace/engram
T=mmlu_anatomy,mmlu_clinical_knowledge,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine
echo ===BASE===
CUDA_VISIBLE_DEVICES=0 $PY examples/eval_mmlu.py --tasks $T --limit 100 --batch_size 16
echo ===ARITH===
CUDA_VISIBLE_DEVICES=0 $PY examples/eval_mmlu.py --engram_weights outputs/benchmarks/ckpt_arithmetic_h8 --tasks $T --limit 100 --batch_size 16
echo ===RQ===
CUDA_VISIBLE_DEVICES=0 $PY examples/eval_mmlu.py --engram_weights outputs/benchmarks/ckpt_rq_h8 --tasks $T --limit 100 --batch_size 16
echo DONE_EVALMATRIX_EXIT_$?
