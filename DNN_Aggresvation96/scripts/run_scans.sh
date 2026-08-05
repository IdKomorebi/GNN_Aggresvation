#!/bin/bash
# 96 号扫描驱动（快速准确率版）：用法 bash scripts/run_scans.sh <shard 0|1> <gpu_id>
#
# 只测**准确率**，不测召回率，故不做全枚举：每阶取均匀随机抽样池，
# 让估计器在池内自选 top-K，再送重训认证。覆盖率如实标注。
#
# 为什么用均匀抽样而不是从 k-1 阶强集合"生长"候选：
#   syn_k(S) = v(S) - max_{T⊂S,|T|=k-1} v(T)。S 若含 v 很高的 (k-1) 阶子集，
#   减数就大、syn 反而小 ⟹ "从强的低阶往上扩"是**反向**启发式。
#   94 号实测：认证为真的五阶协同，其最好四阶父集排名 164-2429，beam B=100 全漏。
#
# k=6 命中已建好的 order-5 全量缓存（子集直接查表，省 7 倍算力），故样本量给大。
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
CK=../DNN_Aggresvation93/outputs/oracle_l1_aug8_seed0.pt
S=$1
export CUDA_VISIBLE_DEVICES=$2

echo "== [1/3] k=6 抽样池（本分片 15 万；子集查表）=="
$PY scripts/scan_highorder.py --ckpt $CK --kind last --order 6 \
    --mode sample --n_sample 150000 --shard $S --nshard 2 --chunk 48

echo "== [2/3] k=7 抽样池（本分片 3 万）=="
$PY scripts/scan_highorder.py --ckpt $CK --kind last --order 7 \
    --mode sample --n_sample 30000 --shard $S --nshard 2 --chunk 48

echo "== [3/3] k=8 抽样池（本分片 3 万）=="
$PY scripts/scan_highorder.py --ckpt $CK --kind last --order 8 \
    --mode sample --n_sample 30000 --shard $S --nshard 2 --chunk 48

echo "== run_scans 分片 $S 全部完成 =="
