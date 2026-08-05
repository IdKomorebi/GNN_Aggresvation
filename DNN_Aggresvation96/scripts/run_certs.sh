#!/bin/bash
# 96 号认证驱动：用法 bash scripts/run_certs.sh <shard 0|1> <gpu_id>
# 每阶 top-10 + 同池随机对照 10（syn<0.05 中随机抽），逐集合重训 1+k 个专用 DNN。
# 协议 = 88 号修正版（训练集内部 val 早停，测试集不参与）。
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
S=$1
export CUDA_VISIBLE_DEVICES=$2
SRC=oracle_l1_aug8_seed0_last

for K in 6 7 8; do
  echo "== k=$K 认证（top-10 + 对照 10，分片 $S/2）=="
  $PY scripts/certify_highorder.py --order $K --source hi_o${K}_${SRC} \
      --n_top 10 --n_ctrl 10 --tag _hi --shard $S --nshard 2
done
echo "== run_certs 分片 $S 全部完成 =="
