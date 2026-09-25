#!/bin/bash
# 等 est_a 结束 → RTS 与 PJM 负荷的读出 → 大集合对照；自写 PID
E=/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation122; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
echo $$ > $E/logs/after_est.pid
P=$(cat $E/logs/est_a.pid); while kill -0 $P 2>/dev/null; do sleep 30; done
while [ ! -f $E/outputs/est/PJM-load/eval_masks.npy ]; do sleep 30; done
$PY $E/scripts/est122.py --jobs 30 --tags RTS-GMLC,PJM-load > $E/logs/est_b.log 2>&1
$PY $E/scripts/large122.py > $E/logs/large.log 2>&1
echo DONE >> $E/logs/est_b.log
