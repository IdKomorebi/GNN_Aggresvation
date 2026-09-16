#!/usr/bin/env bash
# 第二阶段 GPU 流水线（GPU0/1/2），等真值队列（保存的 PID）结束后启动。
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
for f in logs/truth_gpu0.pid logs/truth_gpu1.pid logs/truth_gpu2.pid; do p=$(cat $f); while kill -0 $p 2>/dev/null; do sleep 30; done; done
echo "[$(date +%H:%M:%S)] 真值队列结束，开始第二阶段"
E="$PY scripts/run_est.py"
declare -A TR=([0]="--ds caiso --seed 1" [1]="--ds caiso --seed 2" [2]="--ds caiso --seed 0")
for k in 0 1 2; do
  ( CUDA_VISIBLE_DEVICES=$k $PY scripts/train_uniform.py ${TR[$k]} 2>&1 | grep -E "val=|Error|Traceback"
    for job in "pjm L0ensx" "caiso L0ensx" "pjm L0" "caiso L0" "pjm L1x" "pjm direct" "caiso direct"; do set -- $job
      CUDA_VISIBLE_DEVICES=$k $E --ds $1 --set k4 --model $2 --shard $k --nshard 3 2>&1 | grep -E "^done|exists|Error|Traceback"
    done
    CUDA_VISIBLE_DEVICES=$k $PY scripts/mci_search.py --ds pjm --confs $((k*4)),$((k*4+1)),$((k*4+2)),$((k*4+3)) --tag s$k 2>&1 | grep -E "^done|Error|Traceback"
    CUDA_VISIBLE_DEVICES=$k $PY scripts/mci_search.py --ds caiso --confs $((k*4)),$((k*4+1)),$((k*4+2)),$((k*4+3)) --tag s$k 2>&1 | grep -E "^done|Error|Traceback"
  ) > logs/stage2_gpu$k.log 2>&1 &
  echo $! > logs/stage2_gpu$k.pid
done
wait
$PY scripts/mci_search.py --ds pjm --merge; $PY scripts/mci_search.py --ds caiso --merge
for k in 0 1 2; do
  ( for ds in pjm caiso; do CUDA_VISIBLE_DEVICES=$k $PY scripts/run_truth.py --ds $ds --set mci --seed 0 --shard $k --nshard 3 2>&1 | grep -E "Error|Traceback"; done ) &
done
wait
echo "[$(date +%H:%M:%S)] MCI 认证完成，启动 101 号独立认证"
bash ../DNN_Aggresvation101/scripts/run101.sh > ../DNN_Aggresvation101/logs/run101.log 2>&1
echo "[$(date +%H:%M:%S)] STAGE2_DONE"
