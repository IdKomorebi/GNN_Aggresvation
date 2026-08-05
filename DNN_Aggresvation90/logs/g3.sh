#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation90
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=3 $PY scripts/rescan_full.py --order 5 --kind poly2 --eval audit --shard $s --nshard 2 --chunk 300 >> logs/g3.log 2>&1
done
echo G3_DONE >> logs/g3.log
