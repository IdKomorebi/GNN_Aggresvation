#!/bin/bash
E=/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation122; PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
echo $$ > $E/logs/after_seeds.pid
for P in $(cat $E/logs/seeds.pids) $(cat $E/logs/after_est.pid); do while kill -0 $P 2>/dev/null; do sleep 30; done; done
$PY $E/scripts/est122.py --jobs 30 --variants uniformE,reconE > $E/logs/est_c.log 2>&1
echo DONE >> $E/logs/est_c.log
