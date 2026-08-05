#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation90
CUDA_VISIBLE_DEVICES=1 $PY scripts/rescan_full.py --order 5 --kind full --eval audit --shard 0 --nshard 2 --chunk 300 >> logs/g1.log 2>&1
echo G1_DONE >> logs/g1.log
