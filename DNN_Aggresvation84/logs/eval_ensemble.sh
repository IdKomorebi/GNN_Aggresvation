#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation84
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES="" $PY scripts/eval_designtriples.py \
    --ckpt ../DNN_Aggresvation75/outputs/oracle_uniform_seed${s}.pt \
    --out outputs/eval_uniform_seed${s}.parquet >> logs/eval_ensemble.log 2>&1
done
cp outputs/eval_uniform_seed0.parquet outputs/eval_baseline_uniform.parquet
echo "ENSEMBLE_EVAL_DONE" >> logs/eval_ensemble.log
