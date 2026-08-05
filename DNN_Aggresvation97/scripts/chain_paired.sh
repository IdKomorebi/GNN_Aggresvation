#!/bin/bash
# 硬闸门先行：k=5 复现 93 号的 10 个真协同 → 通过后才跑 k=6 判决。
# 三卡分片（避开 GPU1，其上有他人进程）。
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python

echo "===== 闸门：k=5（21 个 93 号已认证集合，R=3 种子）====="
for i in 0 1 2; do
  g=$([ $i -eq 0 ] && echo 0 || ([ $i -eq 1 ] && echo 2 || echo 3))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/paired_certify.py --order 5 \
     --from_csv "../DNN_Aggresvation93/outputs/certify_o5_l1top_s*.csv" \
     --seeds 0,1,2 --tag _gate --shard $i --nshard 3 > outputs/pair_gate$i.log 2>&1 &
done
wait
echo "===== 闸门跑完，判读 ====="
$PY scripts/analyze_paired.py gate

echo "===== k=6 判决（97 号新 φ 的同一批 20 个集合，R=3 种子）====="
for i in 0 1 2; do
  g=$([ $i -eq 0 ] && echo 0 || ([ $i -eq 1 ] && echo 2 || echo 3))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/paired_certify.py --order 6 \
     --from_csv "../DNN_Aggresvation96/outputs/certify_o6_hi_s*.csv" \
     --seeds 0,1,2 --tag _k6old --shard $i --nshard 3 > outputs/pair_k6_$i.log 2>&1 &
done
wait
echo "★ 配对认证全部完成"
