#!/bin/bash
# 用法：chain.sh <链名> <GPU> "<组 阶段>" ...   —— 在同一 GPU 上依次运行；脚本自己写 PID（避免 $! 取到包装进程）
R=/data1/duhaocun/projects/GNN_Aggresvation; E=$R/DNN_Aggresvation114; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
NAME=$1; GPU=$2; shift 2
echo $$ > $E/logs/chain_$NAME.pid
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for job in "$@"; do
  set -- $job; G=$E/groups/$1
  echo "[$(date +%T)] 开始 $1 $2 (GPU$GPU)"
  $PY $R/DNN_Aggresvation111/scripts/run_stage.py --exp $G --stage $2 --gpu $GPU --jobs 40 > $G/logs/nohup_$2.log 2>&1
  echo "[$(date +%T)] 结束 $1 $2 (exit $?)"
done
echo "[$(date +%T)] CHAIN_DONE"
