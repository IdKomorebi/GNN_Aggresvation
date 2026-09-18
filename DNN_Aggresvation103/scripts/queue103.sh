#!/usr/bin/env bash
# 103 号流水：等全量单目标真值跑完 → CAISO 掩码消融 backbone 补训 → 基线估计（k3 全部集合，输出到本号）
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
for f in logs/truth_gpu0.pid logs/truth_gpu1.pid logs/truth_gpu2.pid; do p=$(cat $f); while kill -0 $p 2>/dev/null; do sleep 60; done; done
echo "[$(date +%H:%M:%S)] 全量单目标真值完成，开始掩码消融 backbone 补训（CAISO none / bern50）"
for k in 0 1; do
  sch=$([ $k -eq 0 ] && echo none || echo bern50)
  CUDA_VISIBLE_DEVICES=$k $PY scripts/train_mask_variant.py --ds caiso --scheme $sch --seed 0 2>&1 | grep -E "val=|Error|Traceback" &
done
wait
echo "[$(date +%H:%M:%S)] 开始基线估计"
for k in 0 1 2; do
  ( for ds in pjm caiso; do for m in lin lin2 direct L0 L0ensx rffx; do
      CUDA_VISIBLE_DEVICES=$k $PY ../DNN_Aggresvation100/scripts/run_est.py --ds $ds --set k3 --model $m --shard $k --nshard 3 --outroot $(pwd) 2>&1 | grep -E "^done|exists|Error|Traceback"
    done; done
    CUDA_VISIBLE_DEVICES=$k $PY ../DNN_Aggresvation100/scripts/run_est.py --ds pjm --set k3 --model L1x --shard $k --nshard 3 --outroot $(pwd) 2>&1 | grep -E "^done|exists|Error|Traceback"
    for ds in pjm caiso; do for sch in none bern50; do
      CUDA_VISIBLE_DEVICES=$k $PY scripts/run_est_mask.py --ds $ds --scheme $sch --shard $k --nshard 3 2>&1 | grep -E "^done|exists|Error|Traceback"
    done; done ) > logs/est_gpu$k.log 2>&1 &
  echo $! > logs/est_gpu$k.pid
done
wait
echo "[$(date +%H:%M:%S)] QUEUE103_DONE"
