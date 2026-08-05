#!/bin/bash
# 95_caiso 扫描链 GPU2：L1 seed1/2 的 o4（跨 seed 过滤用）+ o5/full-o5 的 shard 1/2 + FDR
set -e
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
export CUDA_VISIBLE_DEVICES=2

wait_for() {  # 等 GPU1 写完共享缓存再读（+60s 防写入中读取）
  while [ ! -f "$1" ]; do echo "  等待 $1 ..."; sleep 60; done
  sleep 60
}

echo "== [1/7] subs_o3 cache + o4 scan (L1 seed1) =="
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed1.pt --kind last --order 4 --subs_only
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed1.pt --kind last --order 4 --shard 0 --nshard 1
echo "== [2/7] subs_o3 cache + o4 scan (L1 seed2) =="
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed2.pt --kind last --order 4 --subs_only
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed2.pt --kind last --order 4 --shard 0 --nshard 1
echo "== [3/7] o5 scan (L1 seed0, shard 1/2) =="
wait_for outputs/subs_o4_oracle_l1_aug8_seed0_last.npy
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed0.pt --kind last --order 5 --shard 1 --nshard 2
echo "== [4/7] o5 scan (full, shard 1/2) =="
wait_for outputs/subs_o4_oracle_uniform_seed0_full.npy
$PY scripts/scan_order5.py --ckpt outputs/oracle_uniform_seed0.pt --kind full --order 5 --shard 1 --nshard 2
echo "== [5/7] paired FDR order-3 (cat3, L0 uniform oracle) =="
$PY scripts/paired_fdr.py --order 3 --kind cat3 --chunk 48
echo "== [6/7] o5 subs 缓存 (L1 seed1，供后续跨 seed o5 排名核对) =="
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed1.pt --kind last --order 5 --subs_only
echo "== [7/7] o5 subs 缓存 (L1 seed2) =="
$PY scripts/scan_order5.py --ckpt outputs/oracle_l1_aug8_seed2.pt --kind last --order 5 --subs_only
echo "== scans_gpu2 全部完成 =="
