# -*- coding: utf-8 -*-
"""122 号 步骤 2：CPU 多进程分块读出（每进程单线程）。每块 = (数据组, 变体, 集合区间)。
用法：est122.py --jobs 36 [--tags ...] [--variants ...]"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
VARIANTS = {"random": ["random"], "uniform": ["uniform"], "small": ["small"], "recon": ["recon"], "small_recon": ["small_recon"],
            "uniform+random": ["uniform", "random"], "recon+random": ["recon", "random"],
            # 三种子预测平均（与新主口径 E 对等）：列表的列表 = 每个种子一组特征
            "uniformE": [["uniform@0"], ["uniform@1"], ["uniform@2"]], "reconE": [["recon@0"], ["recon@1"], ["recon@2"]],
            # 推荐配置的种子稳健性（重建主干与随机主干都换种子）
            "recon+random@1": ["recon@1", "random@1"], "recon+random@2": ["recon@2", "random@2"]}
CH = 150
_CACHE = {}


def work(job):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(ROOT, "src")); import m122  # noqa
    sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa
    tag, v, lo, hi = job; T = tag.replace("/", "_")
    if tag not in _CACHE:
        rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; _CACHE[tag] = registry.load_ds(rel)["D"]
    D = _CACHE[tag]; p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]
    masks = np.load(os.path.join(ROOT, "outputs", "est", T, "eval_masks.npy"))[lo:hi]
    def load(k):
        kind, sd = (k.split("@") + ["0"])[:2]; sd = int(sd)
        out = C + p if kind.endswith("recon") else C; mdl = m122.orc.MLPOracle(p, out)
        if kind == "uniform" and sd > 0:   # 现行本组主干的种子 1、2 直接取各号已训练的权重
            rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]
            f = os.path.join(REPO, rel, f"oracle_seed{sd}.pt")
        else:
            f = os.path.join(ROOT, "outputs", "backbones", T, f"{kind}_seed{sd}.pt")
        mdl.load_state_dict(torch.load(f, map_location="cpu", weights_only=True)); return m122.fr.FrozenPhi(mdl.eval(), "last")
    spec_v = VARIANTS[v]
    if isinstance(spec_v[0], list):
        est, sec = m122.readout_avg(D, [[load(k) for k in grp] for grp in spec_v], masks, "cpu")
    else:
        est, sec = m122.readout(D, [load(k) for k in spec_v], masks, "cpu", B=1)
    return job, est, sec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=36)
    ap.add_argument("--tags", default="RTS-GMLC,NEM,PJM-load,PJM-gen/ic,CAISO-load"); ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args(); jobs = []
    for tag in a.tags.split(","):
        T = tag.replace("/", "_"); n = len(np.load(os.path.join(ROOT, "outputs", "est", T, "eval_masks.npy")))
        for v in a.variants.split(","):
            if os.path.exists(os.path.join(ROOT, "outputs", "est", T, f"{v}.npz")):
                continue
            jobs += [(tag, v, lo, min(lo + CH, n)) for lo in range(0, n, CH)]
    print(f"{len(jobs)} 块", flush=True); t0 = time.time(); res = {}
    with Pool(a.jobs) as pool:
        for k, (job, est, sec) in enumerate(pool.imap_unordered(work, jobs)):
            res.setdefault(job[:2], []).append((job[2], est, sec))
            if k % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
            tag, v = job[:2]; T = tag.replace("/", "_")
            n = len(np.load(os.path.join(ROOT, "outputs", "est", T, "eval_masks.npy")))
            if sum(len(e) for _, e, _ in res[(tag, v)]) == n:
                parts = sorted(res.pop((tag, v)), key=lambda x: x[0])
                np.savez(os.path.join(ROOT, "outputs", "est", T, f"{v}.npz"), est=np.concatenate([e for _, e, _ in parts]),
                         sec=float(np.mean([s for _, _, s in parts])))
                print(f"完成 {tag} {v}", flush=True)
    print("全部完成", time.time() - t0)
