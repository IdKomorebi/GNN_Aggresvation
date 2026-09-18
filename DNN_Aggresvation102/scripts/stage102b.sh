#!/usr/bin/env bash
# 102b：等主流水线结束 → 补两个对照（多目标单种子 φ+x,x² 即 L0x；单目标无 φ 的 x+x² ridge），均在 k3 全部集合上
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
p=$(cat logs/stage102.pid); while kill -0 $p 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] 主流水线结束，开始对照组"
for k in 0 1 2; do
  ( for ds in pjm caiso; do
      CUDA_VISIBLE_DEVICES=$k $PY scripts/run_est_multi.py --ds $ds --set k3 --shard $k --nshard 3 2>&1 | grep -E "^done|exists|Error|Traceback"
      for t in $(seq 0 11); do
        CUDA_VISIBLE_DEVICES=$k $PY scripts/run_est_single.py --ds $ds --set k3 --target $t --shard $k --nshard 3 --nophi 2>&1 | grep -E "^done|exists|Error|Traceback"
      done
    done ) > logs/est_b_gpu$k.log 2>&1 &
  echo $! > logs/est_b_gpu$k.pid
done
wait
echo "[$(date +%H:%M:%S)] STAGE102B_DONE"
