#!/bin/bash
# 等各实验的全部阶段结束后自动运行分析。GPU：111 真值在 GPU2；112 真值与两个通用模型在 GPU1；GPU0 他人占用，GPU3 留空。
R=/data1/duhaocun/projects/GNN_Aggresvation; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
for st in multi single oracle; do wait_pid "$(cat $R/DNN_Aggresvation111/logs/$st.pid)"; done
echo "[$(date +%T)] 111 全部阶段结束"
OMP_NUM_THREADS=1 $PY $R/DNN_Aggresvation111/scripts/analyze_exp.py --exp $R/DNN_Aggresvation111 --gain_target "Y_机组出力_223_STEAM_3" > $R/DNN_Aggresvation111/logs/analyze.log 2>&1
echo "[$(date +%T)] 111 分析结束"
for st in multi single oracle; do wait_pid "$(cat $R/DNN_Aggresvation112/logs/$st.pid)"; done
echo "[$(date +%T)] 112 全部阶段结束"
OMP_NUM_THREADS=1 $PY $R/DNN_Aggresvation111/scripts/analyze_exp.py --exp $R/DNN_Aggresvation112 --gain_target "Y_机组出力_PPCCGT" > $R/DNN_Aggresvation112/logs/analyze.log 2>&1
echo "[$(date +%T)] ALLDONE"
