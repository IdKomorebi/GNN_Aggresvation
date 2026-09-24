#!/bin/bash
# 用法：chain.sh <链名> <GPU> "<作业>" ...   在同一 GPU 上依次运行；脚本自己写 PID（避免 $! 取到包装进程）
#   作业格式："truth <组> <single|multi|tree>" 或 "bb <run_bb.py 参数>"
R=/data1/duhaocun/projects/GNN_Aggresvation; E=$R/DNN_Aggresvation116; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
NAME=$1; GPU=$2; shift 2
echo $$ > $E/logs/chain_$NAME.pid
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for job in "$@"; do
  set -- $job
  echo "[$(date +%T)] 开始 $job (GPU$GPU)"
  if [ "$1" = "truth" ]; then
    G=$E/groups/$2
    $PY $R/DNN_Aggresvation111/scripts/run_stage.py --exp $G --stage $3 --gpu $GPU --jobs 40 > $G/logs/nohup_$3.log 2>&1
  else
    shift; $PY $E/scripts/run_bb.py "$@" --gpu $GPU >> $E/logs/nohup_bb_$NAME.log 2>&1
  fi
  echo "[$(date +%T)] 结束 $job (exit $?)"
done
echo "[$(date +%T)] CHAIN_DONE"
