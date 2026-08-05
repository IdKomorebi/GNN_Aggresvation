#!/bin/bash
# 97 号消融网格：把"增广修不好 φ"这一结论的三处缺陷逐个拆开。
# 用法 bash scripts/run_arms.sh <gpu> <arm...>
#
#   r93      93 号设置但修好 support/query 元组错配 —— 隔离"错配"这一项
#   r93x     ★真·复刻 93 号（含元组错配）—— 现役 φ 的对照，应重现 k6≈12
#   noaug    完全不增广的纯 L1 基线
#   rawhi    raw 乘积但阶 3-7、早停计入合成项 —— 单独隔离"阶数+早停"的贡献
#   tanh     tanh 乘积、阶 3-7、早停只看 conf   —— 单独隔离"构造"的贡献
#   tanhboth tanh 乘积、阶 3-7、早停计入合成项 —— ★主候选（三处全修）
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU

for arm in "$@"; do
  case $arm in
    r93)      A="--aug 8 --aug_kind raw  --aug_rmin 2 --aug_rmax 4 --val_mode conf" ;;
    r93x)     A="--aug 8 --aug_kind raw  --aug_rmin 2 --aug_rmax 4 --val_mode conf --aug_pair mismatch" ;;
    noaug)    A="--aug 0                                           --val_mode conf" ;;
    rawhi)    A="--aug 8 --aug_kind raw  --aug_rmin 3 --aug_rmax 7 --val_mode both" ;;
    tanh)     A="--aug 8 --aug_kind tanh --aug_rmin 3 --aug_rmax 7 --val_mode conf" ;;
    tanhboth) A="--aug 8 --aug_kind tanh --aug_rmin 3 --aug_rmax 7 --val_mode both" ;;
    *) echo "未知臂 $arm"; exit 1 ;;
  esac
  echo "== 训练 $arm =="
  $PY scripts/train_l1_hi.py --seed "${SEED:-0}" --tag "_$arm${SEED:+_s$SEED}" $A
done
echo "== GPU$GPU 全部臂完成 =="
