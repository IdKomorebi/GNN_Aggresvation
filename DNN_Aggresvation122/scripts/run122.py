# -*- coding: utf-8 -*-
"""122 号 步骤 1：各主干（训练方式 × 特征拼接）在一个数据组上的训练与诊断（--stage train，GPU）；
读出由 est122.py 在 CPU 上多进程分块完成（GPU 被他人占满，每张卡只剩约 1 GB）。
评估集合：全部规模 ≤3 的集合（M^(2) 所需）+ 至多 1,500 个随机的规模 4 集合（看优势是否随规模增长）。
用法：run122.py --tag PJM-load --gpu 0 [--only uniform,small,...]"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import m122  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--only", default="")
a = ap.parse_args(); dev = f"cuda:{a.gpu}" if a.gpu >= 0 else "cpu"
rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[a.tag]; d = registry.load_ds(rel); D, keys = d["D"], d["keys"]
T = a.tag.replace("/", "_"); OB = os.path.join(ROOT, "outputs", "backbones", T); OE = os.path.join(ROOT, "outputs", "est", T)
os.makedirs(OB, exist_ok=True); os.makedirs(OE, exist_ok=True)
p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]
logf = open(os.path.join(ROOT, "logs", f"run122_{T}.log"), "a")


def say(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {a.tag}: {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()


# 评估集合：规模 ≤3 全部 + 规模 4 随机至多 1500（NEM 的规模 4 集合在 119 号）
k3 = [k for k in keys if len(k) <= 3]
if a.tag == "NEM":
    k4all = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(REPO, "DNN_Aggresvation119/groups/nem_k4/outputs/D.npz"))["keys"]]
else:
    k4all = [k for k in keys if len(k) == 4]
rs = np.random.RandomState(4); k4 = [k4all[i] for i in np.sort(rs.choice(len(k4all), min(1500, len(k4all)), replace=False))]
evalk = k3 + k4; masks = np.zeros((len(evalk), p), np.float32)
for r, k in enumerate(evalk):
    masks[r, list(k)] = 1
json.dump(dict(n3=len(k3), n4=len(k4), keys4=["|".join(map(str, k)) for k in k4]), open(os.path.join(OE, "eval_sets.json"), "w"))
diag_masks = masks[np.random.RandomState(0).choice(len(k3), 120, replace=False)]


def backbone(kind):
    f = os.path.join(OB, f"{kind}_seed0.pt")
    out = C + p if kind.endswith("recon") else C
    if os.path.exists(f):
        mdl = m122.orc.MLPOracle(p, out).to(dev); mdl.load_state_dict(torch.load(f, map_location=dev, weights_only=True)); return mdl.eval()
    if kind == "uniform":   # 与现行本组主干完全相同：直接复用各号已训练的种子 0
        mdl = m122.orc.MLPOracle(p, C).to(dev)
        mdl.load_state_dict(torch.load(os.path.join(d["O"], "oracle_seed0.pt"), map_location=dev, weights_only=True)); mdl.eval()
    else:
        mdl, inf = m122.train(D, kind, 0, dev); say(f"训练 {kind}：{inf}")
    torch.save(mdl.state_dict(), f); return mdl


VARIANTS = {"random": ["random"], "uniform": ["uniform"], "small": ["small"], "recon": ["recon"], "small_recon": ["small_recon"],
            "uniform+random": ["uniform", "random"], "recon+random": ["recon", "random"]}
todo = a.only.split(",") if a.only else list(VARIANTS)
diag = {}
for v in todo:
    if len(VARIANTS[v]) == 1:
        mdl = backbone(VARIANTS[v][0]); diag[v] = m122.diagnose(D, mdl, diag_masks, dev)
        say(f"{v}：失效单元 {diag[v][0]:.3f}，有效维度 {diag[v][1]:.2f}")
    else:
        for k in VARIANTS[v]:
            backbone(k)
json.dump(diag, open(os.path.join(OE, "diagnose.json"), "w"), indent=1)
np.save(os.path.join(OE, "eval_masks.npy"), masks)
say("训练与诊断完成")
