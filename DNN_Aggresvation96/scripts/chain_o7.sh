#!/bin/bash
# k=7 认证编排：GPU0 等 k=7 扫描收尾后立刻起 shard0；GPU1/2/3 等各自的 k=6 认证
# 结束后依次补入 shard1/2/3。四路分片并行，最大化利用空出的卡。
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
SRC=hi_o7_oracle_l1_aug8_seed0_last

run_shard () {   # $1=shard  $2=gpu
  CUDA_VISIBLE_DEVICES=$2 $PY scripts/certify_highorder.py --order 7 \
      --source $SRC --n_top 10 --n_ctrl 10 --tag _hi \
      --shard $1 --nshard 4 > outputs/cert_o7_s$1.log 2>&1
}

# GPU0：等 k=7 扫描两分片齐全
until ls outputs/hi_o7_*_s0of2.parquet >/dev/null 2>&1 \
   && ls outputs/hi_o7_*_s1of2.parquet >/dev/null 2>&1; do sleep 20; done
sleep 10
run_shard 0 0 &

# GPU1/2/3：等对应的 k=6 认证进程退出后补入
for pair in "1 1" "2 2" "3 3"; do
  set -- $pair
  ( while pgrep -f "certify_highorder.py --order 6 .* --shard $(( $1 - 1 )) " >/dev/null; do sleep 20; done
    run_shard $1 $2 ) &
done
wait
echo "== k=7 认证四分片全部完成 =="
