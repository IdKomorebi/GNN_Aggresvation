#!/bin/bash
# 97 号端到端验证：用修好的 φ 在**真实机密字段**上重扫六阶并认证。
#
# 关键设计：复用 96 号的**同一个候选池**（同 sample_seed 960727、同 n_sample、同 nshard），
# 于是与 96 号 k=6 结果构成**配对对照**，唯一变量就是 φ。
# 96 号基线（老 φ）：top 估计 0.102 → 认证 0.031；对照 0.028 → 0.032；无判别力(AUC 0.42)。
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
R96=../DNN_Aggresvation96
CK=$(realpath outputs/oracle_l1_r93_seed0.pt)
STAGE=$1; GPU=$2; SH=$3; NSH=$4
export CUDA_VISIBLE_DEVICES=$GPU
cd $R96
case $STAGE in
  subs)  $PY scripts/scan_highorder.py --ckpt "$CK" --kind last --order 6 \
             --mode build_subs --shard $SH --nshard $NSH --chunk 48 ;;
  scan)  $PY scripts/scan_highorder.py --ckpt "$CK" --kind last --order 6 \
             --mode sample --n_sample 150000 --shard $SH --nshard $NSH --chunk 48 ;;
  cert)  $PY scripts/certify_highorder.py --order 6 \
             --source hi_o6_oracle_l1_r93_seed0_last --n_top 10 --n_ctrl 10 \
             --tag _e2e --shard $SH --nshard $NSH ;;
esac
echo "== $STAGE shard$SH 完成 =="
