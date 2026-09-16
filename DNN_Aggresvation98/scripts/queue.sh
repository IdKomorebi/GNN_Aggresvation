#!/usr/bin/env bash
# 用法：queue.sh GPU "cmd1" "cmd2" ...   在指定 GPU 上顺序执行（本号约定只用 GPU0/1/2）
gpu=$1; shift
for c in "$@"; do
  echo "[$(date +%H:%M:%S)] START $c"
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$c" 2>&1 | grep -E "^done|Error|error|Traceback" 
  echo "[$(date +%H:%M:%S)] END $c"
done
