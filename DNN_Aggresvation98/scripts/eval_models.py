# -*- coding: utf-8 -*-
"""在掩码集上计算各候选"通用集合价值模型" F(S) 的逐目标测试 R²。

候选（全部只用 train；α 在 train 内部 fit/val 选；测试集只报告）：
  lin      原始 X_S 岭回归（零训练，线性闭式 R²）
  lin2     [X_S, X_S²] 岭回归
  poly2    [X_S, X_S², X_iX_j (i<j∈玩家集)]，仅博弈集（玩家 ≤14）
  direct   75 号 uniform oracle 共享读出头直接前向（69/75 号 K=0 口径）
  L0       冻结 75 号主干 last 特征 + 逐集合闭式岭读出（91 号）
  L0x      L0 特征 ⊕ X_S ⊕ X_S²（diantan 的原始通路改进）
  L1       97 号 r93（修错配后的元训练主干）+ 闭式读出
  L1x      L1 ⊕ X_S ⊕ X_S²
  L1ens    r93 三种子特征拼接 ⊕ X_S ⊕ X_S²
  L0ensx   75 号 uniform 三种子特征拼接 ⊕ X_S ⊕ X_S²
  Lmixx    L0(u) ⊕ L1(r93) ⊕ X_S ⊕ X_S²
  mono*    本号单调约束 oracle（λ=0/1/10）直接前向；mono*r 为其主干 + 闭式读出
  rffx     X_S ⊕ X_S² ⊕ 768 维随机傅里叶特征 cos(ω·x_S+b)，ω~N(0,1/|S|)（零训练非参数基，检验"学出的 φ"是否必要）
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation91/src"))
from common import load_pjm, GAME_A, game_b
from featridge import FrozenPhi, ridge_r2, fit_val_idx, load_oracle
from runlog import log

CK = {"u": REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt",
      "u1": REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed1.pt",
      "u2": REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed2.pt",
      "r93_0": REPO / "DNN_Aggresvation97/outputs/oracle_l1_r93_seed0.pt",
      "r93_1": REPO / "DNN_Aggresvation97/outputs/oracle_l1_r93_s1_seed1.pt",
      "r93_2": REPO / "DNN_Aggresvation97/outputs/oracle_l1_r93_s2_seed2.pt"}
for _l in ("0", "1", "10"):
    CK[f"mono{_l}"] = ROOT / f"outputs/oracle_mono{_l}_seed0.pt"
PHIS = {"L0ensx": ["u", "u1", "u2"], "Lmixx": ["u", "r93_0"], "mono0r": ["mono0"], "mono1r": ["mono1"], "mono10r": ["mono10"], "L0": ["u"], "L0x": ["u"], "L1": ["r93_0"], "L1x": ["r93_0"], "L1ens": ["r93_0", "r93_1", "r93_2"]}

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")
D = load_pjm(); g = D["general"]; nG, nC = len(g), len(D["conf"])
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev)
fit_idx, val_idx = fit_val_idx(len(Xtr), dev)
M = np.load(ROOT / f"outputs/masks/{a.set}.npy")
idx = np.arange(len(M))[a.shard::a.nshard]
if a.limit: idx = idx[:a.limit]
players = {"gameA": [g.index(f) for f in GAME_A], "gameB": [g.index(f) for f in game_b(g)]}.get(a.set)

phis = [FrozenPhi(load_oracle(CK[k], nG, nC, dev), "last") for k in PHIS.get(a.model, [])]
direct = (load_oracle(CK["u"], nG, nC, dev) if a.model == "direct" else
          load_oracle(CK[a.model], nG, nC, dev) if a.model in ("mono0", "mono1", "mono10") else None)
dim = {"lin": 44, "lin2": 88, "poly2": 88 + 91, "L0": 256, "L0x": 344, "L1": 256, "L1x": 344, "L1ens": 856, "rffx": 856, "L0ensx": 856, "Lmixx": 600, "mono0r": 256, "mono1r": 256, "mono10r": 256}.get(a.model, 0)
chunk = a.chunk or max(4, int(6e8 / (len(Xtr) * max(dim, 1) * 8)))

def feats(X, m):  # X:(n,nG) m:(B,nG) → (B,n,d)
    xm = X.unsqueeze(0) * m.unsqueeze(1)
    blocks = []
    if a.model in ("lin", "lin2", "poly2", "L0x", "L1x", "L1ens", "rffx", "L0ensx", "Lmixx"):
        blocks.append(xm)
        if a.model != "lin": blocks.append(xm ** 2)
    if a.model == "poly2":
        P = torch.as_tensor(players, device=dev)
        xp = xm[:, :, P]; iu = torch.triu_indices(len(P), len(P), 1, device=dev)
        blocks.append(xp[:, :, iu[0]] * xp[:, :, iu[1]])
    for phi in phis: blocks.append(phi(X, m))
    if a.model == "rffx":
        gen = torch.Generator(device=dev).manual_seed(98)
        W = torch.randn(768, nG, generator=gen, device=dev); bb = torch.rand(768, generator=gen, device=dev) * 6.2832
        scale = 1 / m.sum(1).clamp_min(1).sqrt()                                   # (B,)
        z = torch.einsum("bnk,dk->bnd", xm, W) * scale[:, None, None] + bb
        blocks.append(torch.cos(z) * 1.4142)
    return torch.cat(blocks, 2)

@torch.no_grad()
def run(m):
    if direct is not None:
        out = []
        for b in range(len(m)):
            pred = direct(Xte, m[b:b + 1].expand(len(Xte), -1))
            out.append(1 - ((pred - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0))
        return torch.stack(out).clamp_min(0)
    # float64：精确重复列(ss_mw×3、as_req×2)+小 α 在 float32 下奇异
    return ridge_r2(feats(Xtr, m).double(), Ytr.double(), feats(Xte, m).double(), Yte.double(), fit_idx, val_idx)

od = ROOT / "outputs/est"; od.mkdir(parents=True, exist_ok=True)
f = od / f"{a.set}_{a.model}_s{a.shard}of{a.nshard}.npz"
t0 = time.time(); outs = []
for s in range(0, len(idx), chunk):
    m = torch.as_tensor(M[idx[s:s + chunk]], device=dev)
    r = run(m); r[m.sum(1) == 0] = 0; outs.append(r.float().cpu().numpy())
    if (s // chunk) % 50 == 0: print(f"[{a.set} {a.model} sh{a.shard}] {s+len(m)}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
V = np.concatenate(outs); np.savez(f, idx=idx, v=V)
ms = (time.time() - t0) / len(idx) * 1e3
log("EST", "DONE", f"{a.set} {a.model} sh{a.shard}/{a.nshard} n={len(idx)} {ms:.2f} ms/集合 -> {f.name}", ms_per_set=ms)
print(f"done {a.model} {ms:.2f} ms/set")
