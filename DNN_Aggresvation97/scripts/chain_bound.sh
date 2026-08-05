#!/bin/bash
# E1：六阶全枚举（7,059,052 个集合）——把上界从 4.25% 抽样池推到全空间
# E2：18 个均匀随机六阶集合的重训认证——与估计器无关的分布
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
CK=$(realpath outputs/oracle_l1_r93_seed0.pt)

echo "===== E1：六阶全枚举（三分片，GPU 0/2/3）====="
cd ../DNN_Aggresvation96
for i in 0 1 2; do
  g=$([ $i -eq 0 ] && echo 0 || ([ $i -eq 1 ] && echo 2 || echo 3))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/scan_highorder.py --ckpt "$CK" --kind last \
     --order 6 --mode full --shard $i --nshard 3 --chunk 48 \
     > ../DNN_Aggresvation97/outputs/full_o6_$i.log 2>&1 &
done
wait
echo "===== E1 完成 ====="
cd ../DNN_Aggresvation97

echo "===== E2：18 个均匀随机集合认证（三分片，R=3 种子）====="
for i in 0 1 2; do
  g=$([ $i -eq 0 ] && echo 0 || ([ $i -eq 1 ] && echo 2 || echo 3))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/paired_certify.py --order 6 \
     --from_csv "outputs/uniform_o6_sets.csv" --seeds 0,1,2 \
     --tag _unif --shard $i --nshard 3 > outputs/unif_$i.log 2>&1 &
done
wait
echo "★ E1+E2 全部完成"
