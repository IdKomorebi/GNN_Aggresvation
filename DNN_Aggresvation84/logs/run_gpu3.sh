#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation84
for job in "none 0" "mono 2" "mono 10" "mono_incr 5"; do
  set -- $job
  CUDA_VISIBLE_DEVICES=3 $PY scripts/train_mono.py --penalty $1 --lam $2 --seed 0 >> logs/train_gpu3.log 2>&1
done
echo "GPU3_ALL_DONE" >> logs/train_gpu3.log
