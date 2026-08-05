#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation90
G=$1; shift
for job in "$@"; do
  set -- $job
  CUDA_VISIBLE_DEVICES=$G $PY scripts/rescan_full.py --order $1 --kind $2 --eval audit --permute 7 --chunk 300 >> logs/perm_g$G.log 2>&1
done
echo PERM_G${G}_DONE >> logs/perm_g$G.log
