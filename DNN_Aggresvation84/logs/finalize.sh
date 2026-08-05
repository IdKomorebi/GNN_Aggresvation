#!/bin/bash
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
cd /data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation84
while ! grep -q VARIANTS_EVAL_DONE logs/eval_variants.log 2>/dev/null; do sleep 10; done
CUDA_VISIBLE_DEVICES="" $PY scripts/score_variants.py > outputs/variant_scores.txt 2>&1
CUDA_VISIBLE_DEVICES="" $PY scripts/make_figs.py >> outputs/variant_scores.txt 2>&1
echo "FINALIZE_DONE" >> logs/finalize.log
