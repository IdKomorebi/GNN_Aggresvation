#!/usr/bin/env bash
# 第三阶段：等第二阶段（含 101）结束后，跑 CAISO 对照博弈真值、全量模型 SAGE、效率基准（GPU0 独占计时）。
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
p=$(cat logs/stage2.pid); while kill -0 $p 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] 第二阶段结束，开始第三阶段"
$PY scripts/make_gameC.py
for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k $PY scripts/run_truth.py --ds caiso --set gameC --seed 0 --shard $k --nshard 3 2>&1 | grep -E "Error|Traceback" & done; wait
CUDA_VISIBLE_DEVICES=1 $PY scripts/sage_gameC.py 2>&1 | grep -E "seed|Error|Traceback" &
CUDA_VISIBLE_DEVICES=0 $PY scripts/bench_efficiency.py > logs/bench_efficiency.log 2>&1
wait
echo "[$(date +%H:%M:%S)] STAGE3_DONE"
