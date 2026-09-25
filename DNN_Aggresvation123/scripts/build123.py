# -*- coding: utf-8 -*-
"""123 号 步骤 1：受控机制算例——RTS-GMLC 上的机组报价推断。
与 111 号完全相同的网络、负荷、新能源与分段报价模型（导入 111 号 build_rts.py 的数据与矩阵），唯一改动：
机组 313_CC_1（区域 3，355 MW，全年约 90% 的小时处于部分出力、常为边际机组）的各段报价乘以 (1+κ_t)，
κ_t ~ U(−AMP, AMP) 逐小时独立（环境变量 AMP123；wide=0.3，narrow=0.05）——这是一个私有、随时间变化的报价加成（机组报价在披露规则下为私有信息），作为敏感目标。
机理预期（写在实验之前）：机组处于边际时，本节点电价 λ_g = (1+κ_t)·c_seg(P_g)，于是
  · P_g 单独：只含"出力高低↔加成高低"的弱信息；λ_g 单独：混有负荷水平等其他因素；
  · {P_g, λ_g}：由 P_g 确定所在报价段、再由 λ_g 反解 κ_t —— 强协同；
  · 同区邻近节点电价 λ_near 在不阻塞时 ≈ λ_g，可替代 λ_g；跨频繁阻塞断面的远端电价 λ_far 替代性差；
  · 系统负荷、新能源、阻塞标志只提供"机组是否处于边际"的运行状态背景。"""
import os, sys, time, importlib.util, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import linprog

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
spec = importlib.util.spec_from_file_location("rts111", os.path.join(REPO, "DNN_Aggresvation111/scripts/build_rts.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
U = "313_CC_1"; u = int(R.th.index[R.th["GEN UID"] == U][0]); gseg = R.seg_unit == u
AMP = float(os.environ.get("AMP123", "0.3")); TAG = os.environ.get("TAG123", "wide")
rng = np.random.RandomState(20260925); KAPPA = rng.uniform(-AMP, AMP, R.H)
OUTD = os.path.join(ROOT, "outputs", TAG); os.makedirs(OUTD, exist_ok=True)


def solve(h):
    c = R.c.copy(); off = R.nb; c[off:off + R.ns][gseg] *= (1 + KAPPA[h])
    loadb = R.share @ R.load_rt[h]
    bounds = ([(0, 0) if i == R.REF else (-np.pi, np.pi) for i in range(R.nb)]
              + [(0, x) for x in R.seg_cap] + [(0, float(x)) for x in R.avail[h]] + [(0, float(x)) for x in loadb])
    r = linprog(c, A_ub=R.A_ub, b_ub=R.b_ub, A_eq=R.A_eq, b_eq=loadb, bounds=bounds, method="highs")
    if r.status != 0:
        return None
    x = r.x; pseg = x[R.nb:R.nb + R.ns]; punit = np.bincount(R.seg_unit, weights=pseg, minlength=len(R.th))
    mu = r.ineqlin.marginals
    return dict(h=h, lmp=r.eqlin.marginals, punit=punit, bind=(np.abs(mu[:R.nl]) + np.abs(mu[R.nl:]) > 1e-6), load=float(loadb.sum()))


if __name__ == "__main__":
    t0 = time.time()
    with Pool(40) as pool:
        res = [r for r in pool.map(solve, range(R.H), chunksize=16) if r is not None]
    hrs = np.array([r["h"] for r in res]); LMP = np.array([r["lmp"] for r in res]); PU = np.array([r["punit"] for r in res])
    BIND = np.array([r["bind"] for r in res]); LOAD = np.array([r["load"] for r in res])
    np.savez_compressed(os.path.join(OUTD, "dispatch123.npz"), hours=hrs, LMP=LMP, PU=PU, BIND=BIND, LOAD=LOAD, KAPPA=KAPPA[hrs])
    print(f"求解 {len(res)}/{R.H}，{time.time() - t0:.0f}s；{U} 部分出力小时 {np.mean((PU[:, u] > 1e-3) & (PU[:, u] < R.th.loc[u, 'PMax MW'] - 1e-3)):.3f}")
