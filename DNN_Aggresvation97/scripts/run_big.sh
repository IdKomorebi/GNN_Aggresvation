#!/bin/bash
# 大样本判决：每臂每阶 24 个注入元组（原为 6），三个种子。
# 6 个元组时中位百分位在 1-99 间跳（tanhboth k=6 三种子 95/1/99），撑不起结论。
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
R96=../DNN_Aggresvation96
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
for arm in "$@"; do
  CK=$(realpath $(ls outputs/oracle_l1_${arm}_seed*.pt | head -1))
  echo "== [$arm] n=24 注入判决 =="
  (cd $R96 && $PY scripts/inject_highorder.py --dataset pjm --mode search \
      --est_kind last --ckpt "$CK" --orders 5,6,7 --n_trial 24 \
      --r2s 0.20,0.35 --n_rand 400 --tag "_b${arm}")
done
echo "== GPU$GPU 完成 =="
