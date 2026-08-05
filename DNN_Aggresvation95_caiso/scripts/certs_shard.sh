#!/bin/bash
# 95_caiso 认证驱动：用法 bash scripts/certs_shard.sh <shard: 0|1> <gpu_id>
# 五组认证按序执行，每组 2 分片跨 2 GPU（另一分片由另一实例跑）。
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
S=$1
export CUDA_VISIBLE_DEVICES=$2

echo "== [1/5] o4 L1-top20 + ctrl20 (shard $S/2) =="
$PY scripts/certify_highorder.py --order 4 --source scan_o4_oracle_l1_aug8_seed0_last \
    --n_top 20 --n_ctrl 20 --tag _l1top --shard $S --nshard 2
echo "== [2/5] o5 L1-top21 + ctrl20 (shard $S/2) =="
$PY scripts/certify_highorder.py --order 5 --source scan_o5_oracle_l1_aug8_seed0_last \
    --n_top 21 --n_ctrl 20 --tag _l1top --shard $S --nshard 2
echo "== [3/5] o3 FDR-top10 + ctrl10 (shard $S/2) =="
$PY scripts/certify_highorder.py --order 3 --source fdr_o3_top \
    --n_top 10 --n_ctrl 10 --tag _fdrtop --shard $S --nshard 2
echo "== [4/5] o4 full-top20 (shard $S/2) =="
$PY scripts/certify_highorder.py --order 4 --source scan_o4_oracle_uniform_seed0_full \
    --n_top 20 --n_ctrl 0 --tag _fulltop --shard $S --nshard 2
echo "== [5/5] o5 full-top20 (shard $S/2) =="
$PY scripts/certify_highorder.py --order 5 --source scan_o5_oracle_uniform_seed0_full \
    --n_top 20 --n_ctrl 0 --tag _fulltop --shard $S --nshard 2
echo "== certs_shard $S 全部完成 =="
