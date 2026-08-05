#!/bin/bash
# 等三片子集缓存齐 → 合并 → 双分片扫描六阶（GPU0/2）→ 四分片认证（GPU0/2/3）
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
STEM=oracle_l1_r93_seed0_last
R96=../DNN_Aggresvation96

until [ "$(ls $R96/outputs/subs_o5_${STEM}_part*of3.npy 2>/dev/null | wc -l)" -eq 3 ]; do sleep 20; done
echo "[链] 三片齐，开始合并"
$PY scripts/merge_subs.py $STEM

echo "[链] 起六阶扫描（双分片，GPU0/2）"
bash scripts/run_e2e.sh scan 0 0 2 > outputs/e2e_scan0.log 2>&1 &
bash scripts/run_e2e.sh scan 2 1 2 > outputs/e2e_scan1.log 2>&1 &
wait
echo "[链] 扫描完成，起认证（四分片，GPU 0/2/3/0）"
for pair in "0 0" "1 2" "2 3" "3 0"; do
  set -- $pair
  bash scripts/run_e2e.sh cert $2 $1 4 > outputs/e2e_cert$1.log 2>&1 &
done
wait
echo "[链] ★端到端全部完成"
