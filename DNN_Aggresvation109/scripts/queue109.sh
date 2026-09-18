#!/bin/bash
# 109 号真值队列：2 数据集 × 4 比例，每个任务用 GPU 0/1/2 三卡分片（第 3 张卡留给他人）。
set -u
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
for ds in pjm caiso; do
  for frac in 0.10 0.25 0.50 1.00; do
    tag="${ds}_f$(printf '%03d' $(python3 -c "print(int($frac*100))"))"
    if ls outputs/truth/${tag}_seed0_s*of3.npz >/dev/null 2>&1 && \
       [ "$(ls outputs/truth/${tag}_seed0_s*of3.npz | wc -l)" -eq 3 ]; then
      echo "[skip] $tag 已完成"; continue
    fi
    echo "=== $tag 开始 $(date +%H:%M:%S) ==="
    pids=()
    for sh in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$sh $PY scripts/run_truth_naux.py --ds "$ds" --frac "$frac" \
        --shard $sh --nshard 3 > "logs/${tag}_s${sh}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "=== $tag 结束 $(date +%H:%M:%S) ==="
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
