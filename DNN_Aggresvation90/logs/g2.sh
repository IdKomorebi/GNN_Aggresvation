#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation90
CUDA_VISIBLE_DEVICES=2 $PY scripts/rescan_full.py --order 5 --kind full --eval audit --shard 1 --nshard 2 --chunk 300 >> logs/g2.log 2>&1
echo G2_DONE >> logs/g2.log
