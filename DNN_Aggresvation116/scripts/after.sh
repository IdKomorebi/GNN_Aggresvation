#!/bin/bash
# 用法：after.sh <等待的 PID 文件> <命令...>   等该 PID 退出后再执行命令（用 kill -0 轮询，不用 pgrep）
PF=$1; shift; P=$(cat $PF)
while kill -0 $P 2>/dev/null; do sleep 30; done
exec "$@"
