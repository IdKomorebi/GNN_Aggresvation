#!/bin/bash
# 用法：chain115b.sh <链名> <GPU> <等待的PID或0> <实验目录>...；脚本自写 PID
R=/data1/duhaocun/projects/GNN_Aggresvation; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python; E=$R/DNN_Aggresvation115
NAME=$1; GPU=$2; WAIT=$3; shift 3; echo $$ > $E/logs/chain_$NAME.pid; export OMP_NUM_THREADS=1
[ "$WAIT" != "0" ] && while kill -0 $WAIT 2>/dev/null; do sleep 30; done
for d in "$@"; do
  [ -f $d/outputs/est_variants.npz ] && { echo "[$(date +%T)] 已存在，跳过 $d"; continue; }
  echo "[$(date +%T)] 开始 $d (GPU$GPU)"
  $PY $E/scripts/run_variants.py --exp $d --gpu $GPU > $E/logs/variants_$(basename $d).log 2>&1
  echo "[$(date +%T)] 结束 $d (exit $?)"
done
echo "[$(date +%T)] CHAIN_DONE"
