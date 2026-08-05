#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation84
# 等两条队列都完成
while ! grep -q GPU3_ALL_DONE logs/train_gpu3.log 2>/dev/null || ! grep -q GPU0_ALL_DONE logs/train_gpu0.log 2>/dev/null; do
  sleep 10
done
for v in none_lam0 mono_lam2 mono_lam10 incr_lam2 incr_lam10 mono_incr_lam5; do
  [ -f outputs/eval_${v}.parquet ] && continue
  CUDA_VISIBLE_DEVICES="" $PY scripts/eval_designtriples.py \
    --ckpt outputs/oracle_${v}_seed0.pt --out outputs/eval_${v}.parquet >> logs/eval_variants.log 2>&1
done
echo "VARIANTS_EVAL_DONE" >> logs/eval_variants.log
