#!/bin/bash
# 用法：chain.sh <链名> <GPU> "<组> <阶段> <种子>" ...   同一设备上依次运行；脚本自写 PID
R=/data1/duhaocun/projects/GNN_Aggresvation; E=$R/DNN_Aggresvation119; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
NAME=$1; GPU=$2; shift 2
echo $$ > $E/logs/chain_$NAME.pid
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for job in "$@"; do
  set -- $job
  echo "[$(date +%T)] 开始 $job (dev $GPU)"
  $PY $E/scripts/run_truth119.py --grp $1 --stage $2 --seed $3 --gpu $GPU --jobs 40 >> $E/groups/$1/logs/truth.log 2>&1
  echo "[$(date +%T)] 结束 $job (exit $?)"
done
echo "[$(date +%T)] CHAIN_DONE"
