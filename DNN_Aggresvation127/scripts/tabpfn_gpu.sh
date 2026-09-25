#!/bin/bash
# 用法：tabpfn_gpu.sh <gpu> <分片号>；逐组运行该分片（共 6 片）
for T in RTS-GMLC NEM PJM-load PJM-gen/ic CAISO-load; do
  SCIPY_ARRAY_API=1 OMP_NUM_THREADS=2 /data1/duhaocun/envs/tabpfn_venv/bin/python /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation127/scripts/run_tabpfn.py --gpu $1 --tag $T --shard $2 --nshard 6
done
