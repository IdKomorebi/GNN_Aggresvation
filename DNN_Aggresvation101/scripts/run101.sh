#!/usr/bin/env bash
# 101 号独立认证流水线：对每个 (数据集, rep, half)，在 GPU0/1/2 上并行：
#   1) 半数据上训练 uniform 通用模型 3 个种子  2) L0ensx 估计规模≤3 全部集合  3) 半数据专用重训真值
# 同一 half 的真值既作为 discovery 真值（winner's curse 对照），也作为另一方向的 audit 真值。
cd "$(dirname "$0")/../.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
S=DNN_Aggresvation100/scripts; O=$(pwd)/DNN_Aggresvation101
for ds in pjm caiso; do for rep in 0 1 2; do for h in 0 1; do
  echo "[$(date +%H:%M:%S)] ===== $ds rep$rep half$h"
  for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k $PY $S/train_uniform.py --ds $ds --seed $k --split half --rep $rep --half $h --outroot $O 2>&1 | grep -E "val=|Error|Traceback" & done; wait
  for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k $PY $S/run_est.py --ds $ds --set k3 --model L0ensx --shard $k --nshard 3 --split half --rep $rep --half $h --outroot $O \
       --setfile $(pwd)/DNN_Aggresvation100/outputs/sets/${ds}_k3.npy 2>&1 | grep -E "^done|exists|Error|Traceback" & done; wait
  for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k $PY $S/run_truth_split.py --ds $ds --set k3 --seed 0 --shard $k --nshard 3 --split half --rep $rep --half $h --outroot $O 2>&1 | grep -E "exists|Error|Traceback|3[0-9]{3}/" & done; wait
done; done; done
echo RUN101_DONE
