#!/usr/bin/env bash
# 102 号流水：等 backbone 训练完 → 单目标估计（k3 全部集合）→ 单目标真值抽样诊断（每数据集 1500 个集合 × 12 目标）
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
for f in logs/train_gpu0.pid logs/train_gpu1.pid logs/train_gpu2.pid; do p=$(cat $f); while kill -0 $p 2>/dev/null; do sleep 20; done; done
echo "[$(date +%H:%M:%S)] backbone 训练完成，开始单目标估计"
for k in 0 1 2; do
  ( for ds in pjm caiso; do for t in $(seq 0 11); do
      CUDA_VISIBLE_DEVICES=$k $PY scripts/run_est_single.py --ds $ds --set k3 --target $t --shard $k --nshard 3 2>&1 | grep -E "^done|exists|Error|Traceback"
    done; done ) > logs/est_gpu$k.log 2>&1 &
  echo $! > logs/est_gpu$k.pid
done
wait
echo "[$(date +%H:%M:%S)] 估计完成，开始单目标真值抽样诊断"
for k in 0 1 2; do
  ( for ds in pjm caiso; do for t in $(seq 0 11); do
      CUDA_VISIBLE_DEVICES=$k $PY scripts/truth_single.py --ds $ds --set k3 --target $t --shard $k --nshard 3 --limit 500 --tag _samp 2>&1 | grep -E "^done|exists|Error|Traceback"
    done; done ) > logs/truth_gpu$k.log 2>&1 &
  echo $! > logs/truth_gpu$k.pid
done
wait
echo "[$(date +%H:%M:%S)] STAGE102_DONE"
