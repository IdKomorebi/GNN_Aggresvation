# -*- coding: utf-8 -*-
"""126 号：估计器与对照方法的统一计时。
同一张空闲 GPU、每组同一批 200 个随机规模 ≤3 集合（与 118 号 timeE 相同的随机种子 2）、读出类方法批大小统一为 8；
每个方法先预热一个批次再计时，前后 cuda.synchronize。输出 outputs/time_<组>.csv（方法、每集合毫秒、读出次数、特征维度）。
方法：
  lin / poly / polyS         无主干读出（[x⊙m]；[x⊙m,(x⊙m)²]；集合内二阶字典）
  head1 / head3              共享输出头（只预测目标主干直接前向，单种子 / 三种子平均），不做逐集合读出
  rand1                      随机初始化主干 + 读出（1 次读出）
  D1 / E / Ecat              只预测目标：单种子；三种子预测平均（3 次读出，旧口径）；三种子特征拼接（1 次读出）
  C1 / C2 / C3               重建×3+截断（3 次读出）；重建⊕随机×1+截断（1 次读出）；重建⊕随机×3+截断（3 次读出，本文）
  ft                         热启动微调整个网络（每集合 100 步，每 10 步验证）——只计 20 个集合
用法：bench126.py --gpu 0 [--fast]（--fast：读出换成等价快速实现，输出 time_fast_<组>.csv）"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
from copy import deepcopy

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
B122 = os.path.join(REPO, "DNN_Aggresvation122")
sys.path.insert(0, os.path.join(B122, "src")); import m122  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
orc, fr, ro = m122.orc, m122.fr, m122.ro
FIX2 = dict(sd_floor=1e-3, clip_te=True); FIX3 = dict(FIX2, clip_y=True)
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--tags", default="RTS-GMLC,NEM,PJM-load,PJM-gen/ic,CAISO-load")
ap.add_argument("--fast", action="store_true", help="读出换成等价快速实现 src/ridge_fast.py（结果逐位相同）")
a = ap.parse_args()
if a.fast:
    sys.path.insert(0, os.path.join(ROOT, "src")); import ridge_fast  # noqa: E402
    ro.ridge_predict = ridge_fast.ridge_predict
dev = f"cuda:{a.gpu}"; os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)


def sync():
    torch.cuda.synchronize(dev)


for tag in a.tags.split(","):
    T = tag.replace("/", "_"); rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; d = registry.load_ds(rel); D, keys = d["D"], d["keys"]
    p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]; k3 = [k for k in keys if len(k) <= 3]
    pick = np.random.RandomState(2).choice(len(k3), min(200, len(k3)), replace=False); sets = [k3[i] for i in pick]
    M = np.zeros((len(sets), p), np.float32)
    for r, k in enumerate(sets):
        M[r, list(k)] = 1
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev)
    Ytr = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64); Yte = torch.as_tensor(D["Yte"], device=dev, dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5); Mt = torch.as_tensor(M, device=dev)

    def net(kind, sd):
        if kind == "uniform":
            m = orc.MLPOracle(p, C); f = os.path.join(REPO, rel, f"oracle_seed{sd}.pt")
        else:
            m = orc.MLPOracle(p, C + p if kind == "recon" else C); f = os.path.join(B122, "outputs", "backbones", T, f"{kind}_seed{sd}.pt")
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True)); return m.eval().to(dev)
    PH = {(k, s): fr.FrozenPhi(net(k, s), "last") for k in ("uniform", "recon", "random") for s in range(3)}

    @torch.no_grad()
    def readout_run(groups, fix, raw=True, B=8, Mx=None):
        """groups：每组一个 FrozenPhi 列表 → 一次读出；多组时预测平均。返回每集合秒数与特征维度。"""
        Mx = Mt if Mx is None else Mx; dim = None
        for s in range(0, len(Mx), B):
            m = Mx[s:s + B]; preds = []
            for phis in groups:
                def feats(X):
                    xm = X.unsqueeze(0) * m.unsqueeze(1)
                    return torch.cat(([xm, xm ** 2] if raw else [xm]) + [ph(X, m) for ph in phis], 2).double()
                Ftr = feats(Xtr); dim = Ftr.shape[2]
                preds.append(ro.ridge_predict(Ftr, Ytr, feats(Xte), fi, vi, cv="kfold", folds=folds, **fix))
            ro.r2_from_pred(sum(preds) / len(preds), Yte)
        return dim

    def timed(fn, *args, **kw):
        fn(*args, Mx=Mt[:8], **kw)   # 预热
        sync(); t0 = time.time(); dim = fn(*args, **kw); sync(); return (time.time() - t0) / len(Mt), dim

    @torch.no_grad()
    def polyS_run(Mx=None):
        idx = range(len(Mx)) if Mx is not None else range(len(sets)); S_ = [sets[i] for i in idx]
        Z = Xtr.double(); Zt = Xte.double(); dim = 0
        for k in (1, 2, 3):
            ss = [s for s in S_ if len(s) == k]
            for b in range(0, len(ss), 8):
                st = torch.as_tensor(ss[b:b + 8], device=dev); Ftr = fr.poly2_block(Z, st); dim = max(dim, Ftr.shape[2])
                ro.r2_from_pred(ro.ridge_predict(Ftr, Ytr, fr.poly2_block(Zt, st), fi, vi, cv="kfold", folds=folds, **FIX2), Yte)
        return dim

    @torch.no_grad()
    def head_run(nseeds, Mx=None):
        Mx = Mt if Mx is None else Mx; ms = [net("uniform", s) for s in range(nseeds)]
        for s in range(0, len(Mx), 8):
            m = Mx[s:s + 8]; B = len(m)
            Xb = Xte.unsqueeze(0).expand(B, -1, -1).reshape(-1, p); Mb = m.unsqueeze(1).expand(-1, len(Xte), -1).reshape(-1, p)
            pred = sum(mm(Xb, Mb) for mm in ms).double().reshape(B, len(Xte), C) / nseeds; ro.r2_from_pred(pred, Yte)
        return 0

    def ft_time(nsets=20, steps=100, bs=256):
        base = net("uniform", 0); Yt = torch.as_tensor(D["Ytr"], device=dev); fit = torch.as_tensor(D["fit_idx"], device=dev); val = torch.as_tensor(D["val_idx"], device=dev)
        sync(); t0 = time.time()
        for r in range(nsets):
            m1 = Mt[r:r + 1]; model = deepcopy(base).train(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            g = torch.Generator(device=dev).manual_seed(r)
            def vloss():
                model.eval()
                with torch.no_grad():
                    v = ((model(Xtr[val], m1.expand(len(val), -1)) - Yt[val]) ** 2).mean().item()
                model.train(); return v
            best, bst = vloss(), deepcopy(model.state_dict())
            for st in range(steps):
                ix = fit[torch.randint(len(fit), (bs,), device=dev, generator=g)]
                opt.zero_grad(); ((model(Xtr[ix], m1.expand(bs, -1)) - Yt[ix]) ** 2).mean().backward(); opt.step()
                if (st + 1) % 10 == 0:
                    v = vloss()
                    if v < best:
                        best, bst = v, deepcopy(model.state_dict())
            model.load_state_dict(bst); model.eval()
            with torch.no_grad():
                ro.r2_from_pred(model(Xte, m1.expand(len(Xte), -1)).double()[None], Yte)
        sync(); return (time.time() - t0) / nsets

    U = lambda s: PH[("uniform", s)]; Rc = lambda s: PH[("recon", s)]; Rn = lambda s: PH[("random", s)]
    rows = []
    plan = [("lin", 1, lambda **kw: readout_run([[]], FIX2, raw=False, **kw)),
            ("poly", 1, lambda **kw: readout_run([[]], FIX2, **kw)),
            ("polyS", 1, lambda **kw: polyS_run(**kw)),
            ("head1", 0, lambda **kw: head_run(1, **kw)), ("head3", 0, lambda **kw: head_run(3, **kw)),
            ("rand1", 1, lambda **kw: readout_run([[Rn(0)]], FIX2, **kw)),
            ("D1", 1, lambda **kw: readout_run([[U(0)]], FIX2, **kw)),
            ("E", 3, lambda **kw: readout_run([[U(s)] for s in range(3)], FIX2, **kw)),
            ("Ecat", 1, lambda **kw: readout_run([[U(0), U(1), U(2)]], FIX2, **kw)),
            ("C1", 3, lambda **kw: readout_run([[Rc(s)] for s in range(3)], FIX3, **kw)),
            ("C2", 1, lambda **kw: readout_run([[Rc(0), Rn(0)]], FIX3, **kw)),
            ("C3", 3, lambda **kw: readout_run([[Rc(s), Rn(s)] for s in range(3)], FIX3, **kw))]
    for name, nread, fn in plan:
        sec, dim = timed(fn); rows.append(dict(数据=tag, 方法=name, 每集合ms=sec * 1000, 读出次数=nread, 特征维度=dim))
        print(f"{tag} {name}: {sec * 1000:.1f} ms（维度 {dim}）", flush=True)
    sec = ft_time(); rows.append(dict(数据=tag, 方法="ft", 每集合ms=sec * 1000, 读出次数=0, 特征维度=0)); print(f"{tag} ft: {sec * 1000:.0f} ms", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs", f"time_{'fast_' if a.fast else ''}{T}.csv"), index=False)
    del PH; torch.cuda.empty_cache()
print("全部完成")
