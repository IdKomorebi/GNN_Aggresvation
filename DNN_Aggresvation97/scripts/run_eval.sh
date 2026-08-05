#!/bin/bash
# 97 号判决：对每个训练臂跑两项评估
#   1) 高阶搜索能力 —— 复用 96 号注入测试（元组来自独立 INJ_SEED 流，与训练无交集）
#   2) 真实 conf 尾部保真度 —— 不能为了高阶把主业弄坏
# 用法 bash scripts/run_eval.sh <gpu> <arm...>
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
R96=../DNN_Aggresvation96
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU

for arm in "$@"; do
  CK=$(realpath $(ls outputs/oracle_l1_${arm}_seed*.pt | head -1))
  echo "== [$arm] 高阶注入搜索 =="
  (cd $R96 && $PY scripts/inject_highorder.py --dataset pjm --mode search \
      --est_kind last --ckpt "$CK" --orders 5,6,7 --n_trial ${NTRIAL:-6} \
      --r2s 0.20,0.35 --n_rand 400 --tag "_a${arm}${NTRIAL:+_n$NTRIAL}")
  echo "== [$arm] 真实 conf 二阶尾部 =="
  $PY scripts/eval_l1_order2.py --ckpt "$CK" --kinds last --tag "_a${arm}"
done
echo "== GPU$GPU 评估完成 =="
