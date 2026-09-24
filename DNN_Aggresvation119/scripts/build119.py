# -*- coding: utf-8 -*-
"""119 号 步骤 1：稳健性与 K=3 证据所需的数据组。
  rts_seed1 / rts_seed2        : RTS-GMLC（111 号）同一划分，真值攻击器换随机种子（DNN 初始化与小批顺序、树的早停划分）
  rts_split43 / rts_split44    : RTS-GMLC 换一次随机 70/30 划分（随机打乱，与时间无关），真值全部重训
  pjm_load_seed1 / _seed2      : PJM 实际负荷（116 号补全候选后）同一划分，真值换随机种子
  nem_k4                       : NEM（112 号）补做规模为 4 的全部集合（112 号只到 3），用于 K=3 的正式口径
"""
import os, sys, json, shutil
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402


def mk(g):
    for s in ("outputs", "logs"):
        os.makedirs(os.path.join(ROOT, "groups", g, s), exist_ok=True)
    return os.path.join(ROOT, "groups", g, "outputs")


for g, src in [("rts_seed1", "DNN_Aggresvation111/outputs"), ("rts_seed2", "DNN_Aggresvation111/outputs"),
               ("pjm_load_seed1", "DNN_Aggresvation116/groups/pjm_load/outputs"), ("pjm_load_seed2", "DNN_Aggresvation116/groups/pjm_load/outputs")]:
    O = mk(g)
    for f in ("D.npz", "fields.json"):
        shutil.copy(os.path.join(REPO, src, f), os.path.join(O, f))
    print(g, "← 复制", src)
spec = json.load(open(os.path.join(REPO, "DNN_Aggresvation111/outputs/fields.json"), encoding="utf-8"))
df = pd.read_csv(os.path.join(REPO, "DNN_Aggresvation111/outputs/dataset.csv"))
for sd in (43, 44):
    O = mk(f"rts_split{sd}"); pipe.SPLIT_SEED = sd
    D = pipe.prep(df, spec["cand"], spec["targ"]); masks, keys = pipe.enumerate_sets(len(spec["cand"]), spec["kmax"])
    np.savez(os.path.join(O, "D.npz"), **{k: v for k, v in D.items() if isinstance(v, np.ndarray)},
             masks=masks, keys=np.array(["|".join(map(str, k)) for k in keys]))
    json.dump(dict(spec, split_seed=sd), open(os.path.join(O, "fields.json"), "w"), ensure_ascii=False, indent=1)
    print(f"rts_split{sd}：train {len(D['Xtr'])} / test {len(D['Xte'])}，集合 {len(keys)}")
pipe.SPLIT_SEED = 42
O = mk("nem_k4"); z = np.load(os.path.join(REPO, "DNN_Aggresvation112/outputs/D.npz"))
spec = json.load(open(os.path.join(REPO, "DNN_Aggresvation112/outputs/fields.json"), encoding="utf-8"))
p = len(spec["cand"]); masks, keys = pipe.enumerate_sets(p, 4); s4 = np.array([len(k) == 4 for k in keys])
np.savez(os.path.join(O, "D.npz"), **{k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]},
         masks=masks[s4], keys=np.array(["|".join(map(str, k)) for k, t in zip(keys, s4) if t]))
json.dump(dict(spec, kmax=4, note="只含规模为 4 的集合；规模 ≤3 的真值取自 112 号"), open(os.path.join(O, "fields.json"), "w"), ensure_ascii=False, indent=1)
print("nem_k4：规模 4 的集合", int(s4.sum()))
