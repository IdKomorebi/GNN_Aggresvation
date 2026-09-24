# -*- coding: utf-8 -*-
"""118 号：通用推断模型的对比与消融（规模 ≤3 的全部集合，K≤2 所需）。每个变体输出每集合每目标的测试 R² 估计。

对比（不用预训练主干或换一种估计方式）：
  lin     线性读出：[x⊙m]                                    —— 线性攻击者
  poly    [x⊙m, (x⊙m)²]                                       —— 无主干的手工特征
  polyS   集合内二阶多项式：S 内字段的原值、平方、两两乘积     —— 91 号 poly2 字典
  head3   共享输出头：主干直接输出预测（三种子预测平均），不做子集读出
  ft      热启动微调：从主干出发，对每个集合单独微调整个网络（最多 100 步，val 早停）——只在 150 个随机集合上做
消融（单种子，读出设置同新主口径：数值修正 + 5 折选 λ）：
  phionly1   只用 φ，不拼 [x⊙m, (x⊙m)²]
  rand1      主干不训练（随机初始化）——检验"是预训练起作用，还是只是随机非线性特征"
  masknone1  预训练不做随机掩码（永远全可见）
  maskbern1  预训练用伯努利掩码（每字段独立以 0.5 可见），而非"先抽可见数再均匀抽"
  nofix1     不做数值修正（标准差下限 1e-8、单次 fit/val 选 λ、不截断）——115 号缺陷的复现
已有、直接复用：D（单种子 + 修正）与 E（三种子预测平均，新主口径）——111/112 号取自 115 号 est_variants.npz，116 号取自 est_group.npz。
另测：逐个串行重训一个集合（单目标 DNN + 两种配置的梯度提升树）所需时间，用于成本对照。
数据边界：预训练、读出、微调都只用训练行（早停与 λ 选择用 train 内部 val）；测试行只用于报告 R²。
用法：run118.py --tag PJM-load --gpu 0 [--only v1,v2]
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, torch
from copy import deepcopy

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation116", "src")); import bb116  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
orc, fr, ro = bb116.orc, bb116.fr, bb116.ro
FIX = dict(sd_floor=1e-3, clip_te=True)

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--only", default="")
a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[a.tag]
d = registry.load_ds(rel); O = d["O"]; D, spec, keys = d["D"], d["spec"], d["keys"]
OUT = os.path.join(ROOT, "outputs", "est", a.tag.replace("/", "_")); os.makedirs(OUT, exist_ok=True)
s3 = np.array([len(k) <= 3 for k in keys]); keys3 = [k for k, t in zip(keys, s3) if t]
p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]
masks = np.zeros((len(keys3), p), np.float32)
for r, k in enumerate(keys3):
    masks[r, list(k)] = 1
Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev)
Ytr = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64); Yte = torch.as_tensor(D["Yte"], device=dev, dtype=torch.float64)
n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
logf = open(os.path.join(ROOT, "logs", f"run118_{a.tag.replace('/', '_')}.log"), "a")


def say(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {a.tag}: {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()


def oracle(seed, kind="trained"):
    m = orc.MLPOracle(p, C).to(dev)
    if kind == "trained":
        m.load_state_dict(torch.load(os.path.join(O, f"oracle_seed{seed}.pt"), map_location=dev, weights_only=True))
    elif kind in ("masknone", "maskbern"):
        m.load_state_dict(torch.load(os.path.join(OUT, f"oracle_{kind}_seed{seed}.pt"), map_location=dev, weights_only=True))
    else:
        torch.manual_seed(1000 + seed); m = orc.MLPOracle(p, C).to(dev)   # 随机初始化，不训练
    return m.eval()


def train_variant(kind, seed=0, epochs=400, patience=60, bs=256):
    """与 pipe.train_oracle 完全相同，只换掩码分布。"""
    torch.manual_seed(seed); np.random.seed(seed)
    Yt = torch.as_tensor(D["Ytr"], device=dev); fit = torch.as_tensor(D["fit_idx"], device=dev); val = torch.as_tensor(D["val_idx"], device=dev)
    model = orc.MLPOracle(p, C).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    mrng = np.random.RandomState(1234 + seed)

    def samp(b, r):
        if kind == "masknone":
            return np.ones((b, p), np.float32)
        m = (r.rand(b, p) < 0.5).astype(np.float32); m[m.sum(1) == 0, 0] = 1; return m
    vr = np.random.RandomState(999); vms = [torch.as_tensor(orc.sample_mask(len(val), p, vr), device=dev) for _ in range(8)]
    best, bst, pat = 1e9, None, 0
    for ep in range(epochs):
        model.train(); order = fit[torch.randperm(len(fit), device=dev)]
        for b in range(0, len(order), bs):
            ix = order[b:b + bs]; m = torch.as_tensor(samp(len(ix), mrng), device=dev)
            opt.zero_grad(); ((model(Xtr[ix], m) - Yt[ix]) ** 2).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad():   # 早停用同一组"均匀可见数"验证掩码，与主口径可比
            v = float(np.mean([((model(Xtr[val], mm) - Yt[val]) ** 2).mean().item() for mm in vms]))
        if v < best - 1e-6:
            best, bst, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(bst); torch.save(model.state_dict(), os.path.join(OUT, f"oracle_{kind}_seed{seed}.pt"))
    return ep + 1


@torch.no_grad()
def readout(feat_fn, B=8, **kw):
    out, t0 = [], time.time(); Mt = torch.as_tensor(masks, device=dev)
    for s in range(0, len(Mt), B):
        m = Mt[s:s + B]
        pred = ro.ridge_predict(feat_fn(Xtr, m), Ytr, feat_fn(Xte, m), fi, vi, **kw)
        out.append(ro.r2_from_pred(pred, Yte))
    return torch.cat(out).cpu().numpy().astype(np.float32), (time.time() - t0) / len(Mt)


def xm(X, m):
    return X.unsqueeze(0) * m.unsqueeze(1)


KF = dict(cv="kfold", folds=folds, **FIX)
variants = {
    "lin": lambda: readout(lambda X, m: xm(X, m).double(), B=32, **KF),
    "poly": lambda: readout(lambda X, m: torch.cat([xm(X, m), xm(X, m) ** 2], 2).double(), B=32, **KF),
}


def v_polyS():
    out = np.zeros((len(keys3), C), np.float32); t0 = time.time()
    Z = torch.as_tensor(D["Xtr"], device=dev, dtype=torch.float64); Zt = torch.as_tensor(D["Xte"], device=dev, dtype=torch.float64)
    for k in (1, 2, 3):
        rows = [r for r, kk in enumerate(keys3) if len(kk) == k]
        for s in range(0, len(rows), 64):
            rr = rows[s:s + 64]; sets = torch.as_tensor([keys3[r] for r in rr], device=dev)
            pred = ro.ridge_predict(fr.poly2_block(Z, sets), Ytr, fr.poly2_block(Zt, sets), fi, vi, **KF)
            out[rr] = ro.r2_from_pred(pred, Yte).cpu().numpy()
    return out, (time.time() - t0) / len(keys3)


def v_phi(kind, raw=True, fix=True):
    ph = fr.FrozenPhi(oracle(0, kind), "last")
    f = (lambda X, m: torch.cat([xm(X, m), xm(X, m) ** 2, ph(X, m)], 2).double()) if raw else (lambda X, m: ph(X, m).double())
    return readout(f, **(KF if fix else dict(cv="holdout")))


@torch.no_grad()
def v_head():
    ms = [oracle(s) for s in range(3)]; out, t0 = [], time.time(); Mt = torch.as_tensor(masks, device=dev)
    for s in range(0, len(Mt), 64):
        m = Mt[s:s + 64]; B = len(m)
        Xb = Xte.unsqueeze(0).expand(B, -1, -1).reshape(-1, p); Mb = m.unsqueeze(1).expand(-1, len(Xte), -1).reshape(-1, p)
        pred = sum(mm(Xb, Mb) for mm in ms).double().reshape(B, len(Xte), C) / 3
        out.append(ro.r2_from_pred(pred, Yte))
    return torch.cat(out).cpu().numpy().astype(np.float32), (time.time() - t0) / len(Mt)


def v_ft(nsets=150, steps=100, bs=256):
    """热启动微调：对每个集合，从主干（种子 0）复制一份，固定该集合的掩码训练整个网络，每 10 步在 val 上检查，取 val 最优。"""
    rng = np.random.RandomState(0); pick = np.sort(rng.choice(len(keys3), min(nsets, len(keys3)), replace=False))
    base = oracle(0); Yt = torch.as_tensor(D["Ytr"], device=dev); fit = torch.as_tensor(D["fit_idx"], device=dev); val = torch.as_tensor(D["val_idx"], device=dev)
    out = np.zeros((len(pick), C), np.float32); t0 = time.time()
    for j, r in enumerate(pick):
        m1 = torch.as_tensor(masks[r], device=dev)[None]; model = deepcopy(base).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4); g = torch.Generator(device=dev).manual_seed(int(r))
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
            pred = model(Xte, m1.expand(len(Xte), -1)).double()
        out[j] = ro.r2_from_pred(pred[None], Yte).cpu().numpy()[0]
    return out, (time.time() - t0) / len(pick), pick


def v_seq(nsets=10):
    """逐个串行重训的单集合耗时：单目标 DNN（逐目标一个网络）+ 两种配置的梯度提升树（单线程）。"""
    bt = pipe._load("bt98", pipe.REPO / "DNN_Aggresvation98/src/batch_truth.py")
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.RandomState(1); pick = rng.choice(len(keys3), nsets, replace=False); tt = []
    for r in pick:
        t0 = time.time()
        for c in range(C):
            Dc = dict(D); Dc["Ytr"] = D["Ytr"][:, [c]]; Dc["Yte"] = D["Yte"][:, [c]]
            bt.train_batched(masks[r:r + 1], Dc, seed=0, device=dev)
        t_dnn = time.time() - t0; t1 = time.time(); s = list(keys3[r])
        for c in range(C):
            for lr, leaf in [(0.05, 15), (0.1, 31)]:
                HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaf, max_iter=300, early_stopping=True,
                                              validation_fraction=0.15, random_state=0).fit(D["Xtr"][D["fit_idx"]][:, s], D["Ytr"][D["fit_idx"], c])
        tt.append((t_dnn, time.time() - t1))
    return np.array(tt)


todo = a.only.split(",") if a.only else ["lin", "poly", "polyS", "head3", "phionly1", "rand1", "nofix1", "masknone1", "maskbern1", "ft", "seq"]
for v in todo:
    f = os.path.join(OUT, f"{v}.npz")
    if os.path.exists(f):
        say(f"{v} 已存在，跳过"); continue
    t0 = time.time()
    if v in variants:
        est, sec = variants[v](); np.savez(f, est=est, sec=sec)
    elif v == "polyS":
        est, sec = v_polyS(); np.savez(f, est=est, sec=sec)
    elif v == "head3":
        est, sec = v_head(); np.savez(f, est=est, sec=sec)
    elif v == "phionly1":
        est, sec = v_phi("trained", raw=False); np.savez(f, est=est, sec=sec)
    elif v == "rand1":
        est, sec = v_phi("random"); np.savez(f, est=est, sec=sec)
    elif v == "nofix1":
        ph = fr.FrozenPhi(oracle(0), "last")
        est, sec = readout(lambda X, m: torch.cat([xm(X, m), xm(X, m) ** 2, ph(X, m)], 2).double(), cv="holdout", sd_floor=1e-8)
        np.savez(f, est=est, sec=sec)
    elif v in ("masknone1", "maskbern1"):
        kind = v[:-1]; ep = train_variant(kind, 0); est, sec = v_phi(kind); np.savez(f, est=est, sec=sec, epochs=ep)
    elif v == "ft":
        est, sec, pick = v_ft(); np.savez(f, est=est, sec=sec, pick=pick)
    elif v == "timeE":   # 新主口径 E 的单集合耗时（200 个随机集合，批大小 8；与 seq 在同一张空闲 GPU 上测）
        ms = [oracle(sd) for sd in range(3)]; rng = np.random.RandomState(2); pick = rng.choice(len(keys3), min(200, len(keys3)), replace=False)
        torch.cuda.synchronize(dev); t1 = time.time()
        bb116.estimate_E(D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], ms, masks[pick], list(range(p)), dev)
        torch.cuda.synchronize(dev); sec = (time.time() - t1) / len(pick)
        np.savez(f, E=sec, n3=len(keys3), p=p); say(f"E 单集合耗时 {sec * 1000:.0f} ms")
    elif v == "seq":
        tt = v_seq(); np.savez(f, t=tt); say(f"串行重训单集合：DNN {tt[:, 0].mean():.1f}s + 树 {tt[:, 1].mean():.1f}s")
    say(f"{v} 完成，用时 {time.time() - t0:.0f}s")
say("全部完成")
