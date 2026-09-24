# -*- coding: utf-8 -*-
"""115 号 步骤 2b：主干"一次预训练、跨目标定义复用"。
114 号各组的主干只在本组 15–24 个候选字段、1–2 个目标上预训练。这里改用 75/95/100 号在全部 44 个字段、12 个目标上
预训练好的主干（与 103 号 L0ensx 同一批权重），对 114 号各组的集合做估计：掩码只打开本组候选列，读出只拟合本组目标。
读出设置与变体 E 相同（三种子各自读出后取预测平均；数值修正 + 5 折），记为变体 G。
训练/测试行与 114 号完全一致（同为 100 号 common100.load），测试集只用于 R²。"""
import os, sys, json, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); from readout import ridge_predict, r2_from_pred  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation100", "src")); from common100 import load  # noqa: E402


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


orc = _load("orc69", os.path.join(REPO, "DNN_Aggresvation69/src/oracle.py"))
fr = _load("fr91", os.path.join(REPO, "DNN_Aggresvation91/src/featridge.py"))
dev = torch.device(f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 1}")
CK = {"pjm": [os.path.join(REPO, f"DNN_Aggresvation75/outputs/oracle_uniform_seed{s}.pt") for s in range(3)],
      "caiso": [os.path.join(REPO, "DNN_Aggresvation95_caiso/outputs/oracle_uniform_seed0.pt")]
      + [os.path.join(REPO, f"DNN_Aggresvation100/outputs/oracle_caiso_main_uniform_seed{s}.pt") for s in (1, 2)]}
FIX = dict(sd_floor=1e-3, clip_te=True)
for g, ds in [("pjm_load", "pjm"), ("pjm_gen_ic", "pjm"), ("caiso_load", "caiso")]:
    O = os.path.join(REPO, "DNN_Aggresvation114", "groups", g, "outputs")
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
    keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; sel = np.array([len(k) <= 3 for k in keys])
    D0 = load(ds); gen = D0["general"]; nG, nC = len(gen), D0["Ytr"].shape[1]
    col = [gen.index(c) for c in spec["cand"]]
    assert np.allclose(D0["Xtr"][:, col], z["Xtr"]), "行顺序与 114 号不一致"
    phis = []
    for f in CK[ds]:
        m = orc.MLPOracle(nG, nC).to(dev); ck = torch.load(f, map_location=dev, weights_only=False)
        m.load_state_dict(ck["state"] if "state" in ck else ck); phis.append(fr.FrozenPhi(m.eval(), "last"))
    Xtr = torch.as_tensor(D0["Xtr"], device=dev); Xte = torch.as_tensor(D0["Xte"], device=dev)
    Ytr = torch.as_tensor(z["Ytr"], device=dev, dtype=torch.float64); Yte = torch.as_tensor(z["Yte"], device=dev, dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
    M44 = np.zeros((int(sel.sum()), nG), np.float32)
    for r, k in enumerate([k for k, s in zip(keys, sel) if s]):
        M44[r, [col[i] for i in k]] = 1
    M44 = torch.as_tensor(M44, device=dev); out = []
    for s in range(0, len(M44), 4):
        m = M44[s:s + 4]; preds = []
        for ph in phis:
            xm = lambda X: X.unsqueeze(0) * m.unsqueeze(1)
            Ftr = torch.cat([xm(Xtr), xm(Xtr) ** 2, ph(Xtr, m)], 2).double(); Fte = torch.cat([xm(Xte), xm(Xte) ** 2, ph(Xte, m)], 2).double()
            preds.append(ridge_predict(Ftr, Ytr, Fte, fi, vi, cv="kfold", folds=folds, **FIX))
        out.append(r2_from_pred(sum(preds) / len(preds), Yte))
    G = torch.cat(out).cpu().numpy().astype(np.float32)
    np.savez(os.path.join(O, "est_G.npz"), sel=sel, G=G)   # 独立文件，避免与 run_variants.py 的输出竞写
    print(f"{g}: 变体 G 完成 {len(G)} 个集合", flush=True)
