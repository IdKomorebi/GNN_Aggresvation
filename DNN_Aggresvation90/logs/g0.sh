#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation90
for k in poly2 full; do
  for o in 3 4; do
    CUDA_VISIBLE_DEVICES=0 $PY scripts/rescan_full.py --order $o --kind $k --eval audit --chunk 400 >> logs/g0.log 2>&1
  done
done
echo G0_DONE >> logs/g0.log
