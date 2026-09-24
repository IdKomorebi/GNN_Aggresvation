#!/bin/bash
# 用法：chain118.sh <链名> <GPU> <变体列表> <数据标签>...   依次运行；脚本自写 PID
R=/data1/duhaocun/projects/GNN_Aggresvation; E=$R/DNN_Aggresvation118; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
NAME=$1; GPU=$2; ONLY=$3; shift 3
echo $$ > $E/logs/chain_$NAME.pid
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for tag in "$@"; do
  echo "[$(date +%T)] 开始 $tag (GPU$GPU)"
  $PY $E/scripts/run118.py --tag "$tag" --gpu $GPU --only $ONLY >> $E/logs/nohup_$NAME.log 2>&1
  echo "[$(date +%T)] 结束 $tag (exit $?)"
done
echo "[$(date +%T)] CHAIN_DONE"
