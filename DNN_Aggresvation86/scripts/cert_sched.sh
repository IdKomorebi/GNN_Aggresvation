#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation86
GPU=$1; SHARD=$2; NSHARD=$3
ids=$($PY -c "import json;s=sorted(json.load(open('outputs/subsets.json')));[print(x) for i,x in enumerate(s) if i%$NSHARD==$SHARD]")
for sid in $ids; do
  [ -f outputs/retrain/${sid}_dnn_seed0.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/retrain_worker.py --subset-id $sid --struct dnn --seed 0 >> logs/cert_gpu${GPU}.log 2>&1
done
echo "CERT_GPU${GPU}_DONE" >> logs/cert_gpu${GPU}.log
