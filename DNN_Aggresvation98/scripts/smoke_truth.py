"""冒烟：并行真值引擎测速 + 与 69 号历史真值（同为 leak69 口径）对照。"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_pjm
from batch_truth import train_batched, ridge_truth
D = load_pjm(); g = D["general"]; dev = torch.device("cuda")
tl = pd.read_csv(ROOT.parent / "DNN_Aggresvation69/outputs/truth_long.csv")
subs = json.load(open(ROOT.parent / "DNN_Aggresvation69/outputs/subsets.json"))
sids = [s for s in tl.canonical_sid.unique()][:256]
masks = np.zeros((len(sids), 44), np.float32)
for b, s in enumerate(sids):
    masks[b, [g.index(f) for f in subs[s]["fields"]]] = 1
t = time.time(); R = train_batched(masks, D, seed=0, device=dev); torch.cuda.synchronize()
print("B=256 time", time.time() - t, "epochs", R["epochs"])
t = time.time(); L = ridge_truth(masks, D, dev); print("ridge time", time.time() - t)
H = tl.drop_duplicates(["canonical_sid", "conf"]).pivot(index="canonical_sid", columns="conf", values="dnn").loc[sids, D["conf"]].values
for k in ["clean", "leak69", "oracle_best"]:
    v = np.clip(R[k], 0, None)
    print(k, "mean", v.mean().round(4), "MAE vs hist69", np.abs(v - H).mean().round(4), "corr", np.corrcoef(v.ravel(), H.ravel())[0, 1].round(4))
print("hist69 mean", H.mean().round(4), " ridge mean", np.clip(L, 0, None).mean().round(4))
print("leak69 - clean mean", (np.clip(R['leak69'],0,None) - np.clip(R['clean'],0,None)).mean().round(4))
