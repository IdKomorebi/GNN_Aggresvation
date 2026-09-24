#!/bin/bash
# 115 号 GPU2 队列：依次对各实验跑读出变体；CAISO 组需等 114 号 GPU2 链（其通用模型）结束。脚本自写 PID。
R=/data1/duhaocun/projects/GNN_Aggresvation; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
E=$R/DNN_Aggresvation115; echo $$ > $E/logs/chain115.pid; export OMP_NUM_THREADS=1
run() { echo "[$(date +%T)] 开始 $1"; $PY $E/scripts/run_variants.py --exp $1 --gpu 2 > $E/logs/variants_$(basename $1).log 2>&1; echo "[$(date +%T)] 结束 $1 (exit $?)"; }
run $R/DNN_Aggresvation112
run $R/DNN_Aggresvation111
run $R/DNN_Aggresvation114/groups/pjm_load
run $R/DNN_Aggresvation114/groups/pjm_gen_ic
W=$(cat $R/DNN_Aggresvation114/logs/chain_gpu2.pid); while kill -0 $W 2>/dev/null; do sleep 30; done
run $R/DNN_Aggresvation114/groups/caiso_load
echo "[$(date +%T)] CHAIN_DONE"
