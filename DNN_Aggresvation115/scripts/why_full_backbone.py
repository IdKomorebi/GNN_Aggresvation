# -*- coding: utf-8 -*-
"""115 号 步骤 4：为什么"全字段全目标主干"让 V 更准、M 只持平？
在 114 号三组上比较两种主干（E：本组 1–2 个目标、15–24 个字段上预训练；G：全部 12 个目标、44 个字段上预训练），
同一批本组字段掩码下：
  (1) 表征：φ 的死神经元比例（训练集上标准差 <1e-6 的单元）、有效维度（协方差特征值的参与比 (Σλ)²/Σλ²）；
  (2) 误差分解：V 的带符号偏差（系统性低估）vs 全部边际 Δ(T,i) 的误差——前者在差分中会相互抵消。"""
import os, sys, json, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation100", "src")); from common100 import load  # noqa: E402


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


orc = _load("orc69", os.path.join(REPO, "DNN_Aggresvation69/src/oracle.py"))
fr = _load("fr91", os.path.join(REPO, "DNN_Aggresvation91/src/featridge.py"))
dev = torch.device(f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 1}")
CK = {"pjm": [os.path.join(REPO, f"DNN_Aggresvation75/outputs/oracle_uniform_seed{s}.pt") for s in range(3)],
      "caiso": [os.path.join(REPO, "DNN_Aggresvation95_caiso/outputs/oracle_uniform_seed0.pt")]
      + [os.path.join(REPO, f"DNN_Aggresvation100/outputs/oracle_caiso_main_uniform_seed{s}.pt") for s in (1, 2)]}
rows = []
for g, ds in [("pjm_load", "pjm"), ("pjm_gen_ic", "pjm"), ("caiso_load", "caiso")]:
    O = os.path.join(REPO, "DNN_Aggresvation114", "groups", g, "outputs")
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
    keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; sel = np.array([len(k) <= 3 for k in keys])
    keys3 = [k for k, s in zip(keys, sel) if s]
    D0 = load(ds); gen = D0["general"]; col = [gen.index(c) for c in spec["cand"]]; p = len(col)
    rs = np.random.RandomState(0); pick = rs.choice(len(keys3), min(300, len(keys3)), replace=False)
    # ---------- (1) 表征
    for tag in ["E 本组主干", "G 全字段主干"]:
        if tag.startswith("E"):
            models = []
            for s in range(3):
                m = orc.MLPOracle(p, len(spec["targ"])).to(dev)
                m.load_state_dict(torch.load(os.path.join(O, f"oracle_seed{s}.pt"), map_location=dev, weights_only=True)); models.append(m.eval())
            X = torch.as_tensor(z["Xtr"], device=dev); width = p; cmap = list(range(p))
        else:
            models = []
            for f in CK[ds]:
                m = orc.MLPOracle(len(gen), 12).to(dev); ck = torch.load(f, map_location=dev, weights_only=False)
                m.load_state_dict(ck["state"] if "state" in ck else ck); models.append(m.eval())
            X = torch.as_tensor(D0["Xtr"], device=dev); width = len(gen); cmap = col
        dead, pr = [], []
        for m in models:
            ph = fr.FrozenPhi(m, "last")
            for r in pick:
                mk = torch.zeros(1, width, device=dev); mk[0, [cmap[i] for i in keys3[r]]] = 1
                F = ph(X, mk)[0].double()                                   # (n,256)
                sd = F.std(0); dead.append(float((sd < 1e-6).float().mean()))
                ev = torch.linalg.eigvalsh(torch.cov(F.T)).clamp_min(0)
                pr.append(float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)))
        rows.append(dict(组=g, 主干=tag, 死神经元比例=np.mean(dead), 有效维度=np.mean(pr)))
    # ---------- (2) 误差分解
    V = np.load(os.path.join(O, "V_official.npy"))[sel]
    ests = {"E 本组主干": pipe.closure_max(np.load(os.path.join(O, "est_variants.npz"))["E"], keys3),
            "G 全字段主干": pipe.closure_max(np.load(os.path.join(O, "est_G.npz"))["G"], keys3)}
    _, _, Dt, _, bsz = pipe.m_table(V, keys3, list(range(p)), 2); s2 = bsz <= 2
    for tag, Ve in ests.items():
        De = pipe.m_table(Ve, keys3, list(range(p)), 2)[2]
        r = [x for x in rows if x["组"] == g and x["主干"] == tag][0]
        r.update(V带符号偏差=float((Ve - V).mean()), V绝对误差=float(np.abs(Ve - V).mean()),
                 V误差中系统偏差占比=float(abs((Ve - V).mean()) / np.abs(Ve - V).mean()),
                 边际Δ绝对误差=float(np.abs(De[:, s2] - Dt[:, s2]).mean()))
    print(g, "完成", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "analysis", "115_why_full_backbone.csv"), index=False)
print(R.round(4).to_string(index=False))
