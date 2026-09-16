#!/usr/bin/env bash
# 等真值全部完成后，在 GPU0/1/2 上分片跑博弈集的全部候选模型估计 + SAGE 对照
cd "$(dirname "$0")/.."
PY=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python
until [ $(ls outputs/truth/gameA_seed1_s*of3.npz 2>/dev/null | wc -l) -eq 3 ] && [ -f outputs/oracle_mono10_seed0.pt ] && [ -f outputs/oracle_mono1_seed0.pt ]; do sleep 20; done
E="$PY scripts/eval_models.py"
for k in 0 1 2; do
  cmds=()
  for set in gameB gameA; do
    for m in lin lin2 poly2 direct L0 L0x L1 L1x L1ens L0ensx Lmixx rffx mono0 mono1 mono10 mono0r mono1r mono10r; do
      cmds+=("$E --set $set --model $m --shard $k --nshard 3")
    done
  done
  [ $k -eq 0 ] && cmds+=("$PY scripts/sage_model.py --game gameB" "$PY scripts/sage_model.py --game gameA")
  [ $k -eq 1 ] && cmds+=("for m in mono0 mono1 mono10 mono0r mono1r mono10r; do $E --set rand --model \$m; done")
  scripts/queue.sh $k "${cmds[@]}" > logs/games_gpu$k.log 2>&1 &
done
wait
echo ALLDONE > logs/games_done.flag
