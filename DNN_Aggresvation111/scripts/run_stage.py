# -*- coding: utf-8 -*-
"""通用分阶段运行（111/112 号共用）。实验目录下须有 outputs/dataset.csv 与 outputs/fields.json。

  --stage prep     标准化与划分 → outputs/D.npz
  --stage multi    多目标 DNN 专用重训真值（GPU）   → outputs/truth_multi.npz
  --stage single   逐目标 DNN 专用重训真值（GPU）   → outputs/truth_single.npz
  --stage tree     梯度提升树真值（CPU，单线程多进程） → outputs/truth_tree.npz
  --stage oracle   通用推断模型：3 种子随机掩码预训练 + 闭式读出估计全部集合 → outputs/est.npz
用法：run_stage.py --exp ../DNN_Aggresvation111 --stage multi --gpu 1
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import pipe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True); ap.add_argument("--stage", required=True)
ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--jobs", type=int, default=40)
a = ap.parse_args()
OUT = os.path.join(os.path.abspath(a.exp), "outputs")
spec = json.load(open(os.path.join(OUT, "fields.json"), encoding="utf-8"))
log = open(os.path.join(os.path.abspath(a.exp), "logs", f"stage_{a.stage}.log"), "a")


def say(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {a.stage}: {msg}"
    print(line, flush=True); log.write(line + "\n"); log.flush()


t0 = time.time()
if a.stage == "prep":
    df = pd.read_csv(os.path.join(OUT, "dataset.csv"))
    D = pipe.prep(df, spec["cand"], spec["targ"])
    masks, keys = pipe.enumerate_sets(len(spec["cand"]), spec["kmax"])
    np.savez(os.path.join(OUT, "D.npz"), **{k: v for k, v in D.items() if isinstance(v, np.ndarray)},
             masks=masks, keys=np.array(["|".join(map(str, k)) for k in keys]))
    say(f"样本 {D['n']}，train {len(D['Xtr'])} / test {len(D['Xte'])}，候选 {masks.shape[1]}，集合 {len(keys)}")
    sys.exit(0)

z = np.load(os.path.join(OUT, "D.npz"))
D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
masks = z["masks"]; keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
dev = f"cuda:{a.gpu}"

if a.stage in ("multi", "single"):
    say(f"开始，{len(masks)} 个集合，{dev}")
    clean, val = pipe.truth_dnn(D, masks, dev, seed=0, single=(a.stage == "single"))
    np.savez(os.path.join(OUT, f"truth_{a.stage}.npz"), clean=clean, val=val)
elif a.stage == "tree":
    say(f"开始，{len(keys)} 个集合，{a.jobs} 进程")
    clean, val = pipe.truth_tree(D, keys, n_jobs=a.jobs)
    np.savez(os.path.join(OUT, "truth_tree.npz"), clean=clean, val=val)
elif a.stage == "oracle":
    import torch
    models, info = [], []
    for s in range(3):
        m, inf = pipe.train_oracle(D, s, dev); models.append(m); info.append(inf)
        torch.save(m.state_dict(), os.path.join(OUT, f"oracle_seed{s}.pt"))
        say(f"种子 {s}：{inf}")
    est, per = pipe.estimate(D, models, masks, dev, raw=True)
    est1, per1 = pipe.estimate(D, models[:1], masks, dev, raw=True)
    np.savez(os.path.join(OUT, "est.npz"), est=est, est_1seed=est1, per_query_s=per, per_query_s_1seed=per1,
             train_sec=np.array([i["sec"] for i in info]))
    say(f"估计完成：三种子 {per*1000:.1f} ms/集合，单种子 {per1*1000:.1f} ms/集合")
say(f"完成，用时 {time.time()-t0:.0f}s")
