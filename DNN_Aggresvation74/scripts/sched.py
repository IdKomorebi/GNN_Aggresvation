# -*- coding: utf-8 -*-
"""DNN74 调度器：train + eval 成对下发到多张 GPU，跳过已完成的。

GPU 0 常被他人占用（>20GB），默认只用 1/2/3，可用 --gpus 覆盖。

用法：
    python scripts/sched.py --round A                        # Round A：5 变体 × 2 seed
    python scripts/sched.py --round B --base A1_gcn_dynamic  # Round B
    python scripts/sched.py --final --variants B6_all,A1_gcn_dynamic --base A1_gcn_dynamic
"""
import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

R74 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R74 / "src"))
from graph_oracle import VARIANTS, build_round_b            # noqa: E402

PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
LOG = R74 / "logs"
LOG.mkdir(exist_ok=True)


TAG = ""


def done(variant: str, seed: int) -> bool:
    return (R74 / "outputs" / f"est_{variant}{TAG}_seed{seed}.csv").exists()


def launch(variant: str, seed: int, base: str | None, gpu: int):
    """一个 job = 训练 + 评测（串在同一张卡上，避免中间态争抢显存）。"""
    tag = f"{variant}{TAG}_seed{seed}"
    base_arg = f" --base {base}" if base else ""
    tag_arg = f" --tag {TAG}" if TAG else ""
    cmd = (f"{PY} -u {R74}/scripts/train_variant.py --variant {variant} "
           f"--seed {seed}{base_arg}{tag_arg} && "
           f"{PY} -u {R74}/scripts/eval_variant.py --variant {variant} --seed {seed}{tag_arg}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen(["bash", "-c", cmd], env=env,
                         stdout=open(LOG / f"{tag}.log", "w"), stderr=subprocess.STDOUT)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {tag} on GPU {gpu}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=["A", "B"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--variants", default=None, help="逗号分隔，覆盖默认变体表")
    ap.add_argument("--base", default=None, help="Round B / final 所基于的 Round A 胜出者")
    ap.add_argument("--seeds", default=None, help="逗号分隔，默认 A/B 为 0,1，final 为 0,1,2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpus", default="1,2,3")
    args = ap.parse_args()
    global TAG
    TAG = args.tag

    if args.variants:
        variants = args.variants.split(",")
    elif args.round == "A":
        variants = list(VARIANTS)
    elif args.round == "B":
        if not args.base:
            raise SystemExit("Round B 必须给 --base <RoundA 胜出者>")
        variants = list(build_round_b(args.base))
    else:
        raise SystemExit("需要 --round A|B 或 --variants")

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else ([0, 1, 2] if args.final else [0, 1]))
    gpus = [int(g) for g in args.gpus.split(",")]

    jobs = deque((v, s) for v in variants for s in seeds if not done(v, s))
    total = len(jobs)
    print(f"DNN74 调度：{total} 个待跑 job（{len(variants)} 变体 × {len(seeds)} seed，"
          f"已跳过完成的），GPU {gpus}\n", flush=True)
    if not total:
        print("全部已完成。")
        return

    pool, run = list(gpus), {}
    while jobs and pool:
        v, s = jobs.popleft()
        g = pool.pop(0)
        run[g] = (launch(v, s, args.base, g), f"{v}_seed{s}")

    fin_ok = fin_bad = 0
    while run:
        time.sleep(10)
        for g, (p, name) in list(run.items()):
            if p.poll() is None:
                continue
            ok = p.returncode == 0
            fin_ok += ok
            fin_bad += not ok
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {name} "
                  f"[{'OK' if ok else 'FAIL 见 logs/'+name+'.log'}] "
                  f"({fin_ok+fin_bad}/{total})", flush=True)
            del run[g]
            if jobs:
                v, s = jobs.popleft()
                run[g] = (launch(v, s, args.base, g), f"{v}_seed{s}")
            else:
                pool.append(g)

    print(f"\n全部结束：成功 {fin_ok}，失败 {fin_bad}")
    sys.exit(1 if fin_bad else 0)


if __name__ == "__main__":
    main()
