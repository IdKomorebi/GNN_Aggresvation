#!/bin/bash
# 字典对照：同一批注入元组上，比较 φ(last) 与"与目标无关的手工字典"(full) 的搜索端百分位。
# 目的：把"高阶本身难"与"φ 对合成注入信号分布外"分开（91 号 E1 记录的局限）。
# 用法 bash scripts/run_dictctl.sh <dataset> <est_kind> <gpu>
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
export CUDA_VISIBLE_DEVICES=$3
$PY scripts/inject_highorder.py --dataset "$1" --est_kind "$2" \
    --orders 5,6,7,8 --n_trial 6 --mode search --r2s 0.20,0.35 \
    --n_rand 400 --tag _m6
echo "== $1/$2 完成 =="
