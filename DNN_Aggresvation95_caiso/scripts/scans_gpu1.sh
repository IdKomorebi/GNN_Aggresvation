#!/bin/bash
# 95_caiso 扫描链 GPU1：L1 seed0 缓存与扫描 + full 字典缓存与扫描（各取 shard 0/2）
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
export CUDA_VISIBLE_DEVICES=1

L1=outputs/oracle_l1_aug8_seed0.pt
UNI=outputs/oracle_uniform_seed0.pt

echo "== [1/8] subs_o3 cache (L1 seed0) =="
$PY scripts/scan_order5.py --ckpt $L1 --kind last --order 4 --subs_only
echo "== [2/8] subs_o4 cache (L1 seed0) =="
$PY scripts/scan_order5.py --ckpt $L1 --kind last --order 5 --subs_only
echo "== [3/8] o4 scan (L1 seed0, 全量) =="
$PY scripts/scan_order5.py --ckpt $L1 --kind last --order 4 --shard 0 --nshard 1
echo "== [4/8] o5 scan (L1 seed0, shard 0/2) =="
$PY scripts/scan_order5.py --ckpt $L1 --kind last --order 5 --shard 0 --nshard 2
echo "== [5/8] subs_o3 cache (full) =="
$PY scripts/scan_order5.py --ckpt $UNI --kind full --order 4 --subs_only
echo "== [6/8] o4 scan (full, 全量) =="
$PY scripts/scan_order5.py --ckpt $UNI --kind full --order 4 --shard 0 --nshard 1
echo "== [7/8] subs_o4 cache (full) =="
$PY scripts/scan_order5.py --ckpt $UNI --kind full --order 5 --subs_only
echo "== [8/8] o5 scan (full, shard 0/2) =="
$PY scripts/scan_order5.py --ckpt $UNI --kind full --order 5 --shard 0 --nshard 2
echo "== scans_gpu1 全部完成 =="
