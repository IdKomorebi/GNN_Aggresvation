#!/bin/bash
# E3：认证六阶【全枚举】的 top-12（估计 0.15-0.23）——补上 E1 留下的尾巴尖
# 这批候选此前从未认证过：已认证的最强估计仅 0.1195（来自 4.25% 抽样池）。
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
for i in 0 1 2; do
  g=$([ $i -eq 0 ] && echo 0 || ([ $i -eq 1 ] && echo 2 || echo 3))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/paired_certify.py --order 6 \
     --from_csv "outputs/fulltop_o6_sets.csv" --seeds 0,1,2 \
     --tag _fulltop --shard $i --nshard 3 > outputs/fulltop_$i.log 2>&1 &
done
wait
echo "★ E3 全枚举 top-12 认证完成"
