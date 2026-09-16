# -*- coding: utf-8 -*-
"""100 号真值侧分析（不涉及估计器）：
T1 攻击器：DNN vs 梯度提升树（规模≤3），攻击器族取最大（val 选择）
T2 M^(0..3)（DNN，规模≤4 闭包）、M^(0..2)（攻击器族，规模≤3 闭包）、种子噪声（DNN seed0 vs seed1，规模≤3）
T3 升级曲线：增益、档位变化、排名稳定、τ-关键字段数、典型曲线统计
T4 攻击器稳定性：M 绝对值、排序、见证、阈值跨越
T5 PJM 公开基底场景（B = 两个负荷预测）
输出 outputs/analysis/{ds}_*.csv|npz 与 report_truth.md
"""
import sys, glob, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, mk_table, attack_max, index_of
from common100 import BASE_PJM

A = ROOT / "outputs/analysis"; TAUS = [0.5, 0.7, 0.9]; GRADES = [0.05, 0.2, 0.5]
rep = ["# 100 号真值侧分析（自动生成）\n"]


def md(df): return df.to_markdown(floatfmt=".4f")


def load_truth(ds, st, seed, n):
    fs = sorted(glob.glob(str(ROOT / f"outputs/truth/{ds}_{st}_seed{seed}_s*of3.npz")))
    if len(fs) != 3: return None
    out = {k: np.zeros((n, 12), np.float32) for k in ["clean", "val_r2", "ridge"]}
    for f in fs:
        z = np.load(f)
        for k in out: out[k][z["idx"]] = z[k]
    return {k: np.clip(v, -1, 1) for k, v in out.items()}


def grade(x): return np.digitize(x, GRADES)


def kt(a, b):
    return float(np.nanmean([kendalltau(a[:, c], b[:, c])[0] for c in range(a.shape[1])]))


def crit_count(Vbar, keys, active, tau, K, base_val=None):
    """定理 3：τ-关键字段数（逐目标）。"""
    idx = index_of(keys); C = Vbar.shape[1]; cnt = np.zeros(C, int)
    from itertools import combinations
    V0 = base_val if base_val is not None else np.zeros(C)
    for i in active:
        others = [a for a in active if a != i]; hit = np.zeros(C, bool)
        for s in range(K + 1):
            for T in combinations(others, s):
                vT = Vbar[idx[T]] if T else V0
                vTi = Vbar[idx[tuple(sorted(T + (i,)))]]
                hit |= (vT <= tau) & (vTi > tau)
        cnt += hit
    return cnt


def run(ds):
    meta = pickle.load(open(ROOT / f"outputs/sets/{ds}_meta.pkl", "rb"))
    keys4, keys3, gen, conf, act = meta["keys"], meta["keys_k3"], meta["general"], meta["conf"], meta["active"]
    n4, n3 = len(keys4), len(keys3)
    T4 = load_truth(ds, "k4", 0, n4)
    if T4 is None: rep.append(f"## {ds}: k4 真值未完成\n"); return
    k3mask = np.array([len(k) <= 3 for k in keys4])
    res = {}
    # ---------------- T2 DNN M^(0..3)
    Vbar4 = closure(T4["clean"], keys4, Vval=T4["val_r2"])
    M4, W4 = mk_table(Vbar4, keys4, act, K_list=(0, 1, 2, 3))
    res["非单调占比(闭包−原值>0.01)"] = float(((Vbar4 - np.clip(T4["clean"], 0, 1)) > 0.01).mean())
    np.savez(A / f"{ds}_M_dnn.npz", M=M4, W=W4, active=act)
    # 种子噪声
    T3s1 = load_truth(ds, "k3", 1, n3)
    if T3s1 is not None:
        Vb_s1 = closure(T3s1["clean"], keys3, Vval=T3s1["val_r2"]); Ms1, Ws1 = mk_table(Vb_s1, keys3, act, K_list=(0, 1, 2))
        for K in range(3):
            res[f"种子噪声 |ΔM^({K})| 均值"] = float(np.abs(Ms1[:, K] - M4[:, K]).mean())
            res[f"种子噪声 M^({K}) Kendall"] = kt(Ms1[:, K], M4[:, K])
            res[f"种子间见证一致率 K={K}"] = float((Ws1[:, K] >= 0).mean()) if K == 0 else float(np.mean([keys3[a] == keys4[b] if a >= 0 and b >= 0 else a == b for a, b in zip(Ws1[:, K].ravel(), W4[:, K].ravel())]))
    # ---------------- T1 / T4 攻击器
    gz = ROOT / f"outputs/gbr/{ds}_k3.npz"
    if gz.exists():
        g = np.load(gz); d3 = T4["clean"][k3mask]; d3v = T4["val_r2"][k3mask]
        sz = np.array([len(k) for k in keys3])
        rows = []
        for s in [1, 2, 3]:
            k = sz == s
            rows.append(dict(规模=s, DNN均值=np.clip(d3[k], 0, 1).mean(), 树模型均值=np.clip(g["clean"][k], 0, 1).mean(),
                             树模型更强占比_test=(g["clean"][k] > d3[k]).mean(), 树模型被val选中占比=(g["val_r2"][k] > d3v[k]).mean()))
        rep.append(f"## {ds} T1 攻击器对比（规模≤3，逐集合×目标）\n\n" + md(pd.DataFrame(rows).set_index("规模")))
        Vatt, Vatt_val, pick = attack_max([d3, g["clean"]], [d3v, g["val_r2"]])
        Vbar_att = closure(Vatt, keys3, Vval=Vatt_val); Ma, Wa = mk_table(Vbar_att, keys3, act, K_list=(0, 1, 2))
        np.savez(A / f"{ds}_M_attack.npz", M=Ma, W=Wa, active=act)
        st = []
        for K in range(3):
            wd = np.mean([keys3[a] == keys4[b] if a >= 0 and b >= 0 else a == b for a, b in zip(Wa[:, K].ravel(), W4[:, K].ravel())])
            st.append(dict(K=K, M_DNN均值=M4[:, K].mean(), M_攻击器族均值=Ma[:, K].mean(), 平均绝对差=np.abs(Ma[:, K] - M4[:, K]).mean(),
                           最大差=np.abs(Ma[:, K] - M4[:, K]).max(), Kendall=kt(Ma[:, K], M4[:, K]), 见证一致率=wd,
                           档位一致率=(grade(Ma[:, K]) == grade(M4[:, K])).mean()))
        rep.append(f"## {ds} T4 攻击器稳定性：DNN vs DNN∪树模型（val 选择）\n\n" + md(pd.DataFrame(st).set_index("K")))
        idx3 = index_of(keys3)
        tc = []
        for tau in TAUS:
            for K in [0, 1, 2]:
                c_d = crit_count(Vbar4[k3mask], keys3, act, tau, K); c_a = crit_count(Vbar_att, keys3, act, tau, K)
                tc.append(dict(tau=tau, K=K, 关键字段数_DNN=int(c_d.sum()), 关键字段数_攻击器族=int(c_a.sum())))
        rep.append(f"## {ds} τ-关键字段数（12 目标合计，定理 3）\n\n" + md(pd.DataFrame(tc).set_index(["tau", "K"])))
    # ---------------- T3 升级曲线
    prof = []
    for p, i in enumerate(act):
        for c in range(12):
            m = M4[p, :, c]
            prof.append(dict(field=gen[i], conf=conf[c], M0=m[0], M1=m[1], M2=m[2], M3=m[3],
                             W1=",".join(gen[j] for j in keys4[W4[p, 1, c]]) if W4[p, 1, c] >= 0 else "∅",
                             W2=",".join(gen[j] for j in keys4[W4[p, 2, c]]) if W4[p, 2, c] >= 0 else "∅",
                             W3=",".join(gen[j] for j in keys4[W4[p, 3, c]]) if W4[p, 3, c] >= 0 else "∅"))
    P = pd.DataFrame(prof); P.to_csv(A / f"{ds}_profiles.csv", index=False)
    g = P[["M0", "M1", "M2", "M3"]].values
    esc = []
    for K in range(1, 4):
        esc.append(dict(K=K, 平均M=g[:, K].mean(), 平均增益_相对前一阶=(g[:, K] - g[:, K - 1]).mean(),
                        增益大于002占比=((g[:, K] - g[:, K - 1]) > 0.02).mean(), 增益大于01占比=((g[:, K] - g[:, K - 1]) > 0.1).mean(),
                        相对M0升档占比=(grade(g[:, K]) > grade(g[:, 0])).mean(), 相对前一阶升档占比=(grade(g[:, K]) > grade(g[:, K - 1])).mean(),
                        排名Kendall_相对前一阶=kt(M4[:, K], M4[:, K - 1]), 排名Kendall_相对M0=kt(M4[:, K], M4[:, 0]),
                        饱和比_MK除M3=np.mean(g[g[:, 3] > 0.05][:, K] / g[g[:, 3] > 0.05][:, 3])))
    rep.append(f"## {ds} T3 升级曲线（41 字段 × 12 目标 = {len(g)} 条）\n\n" + md(pd.DataFrame(esc).set_index("K")))
    # 典型曲线：按 M0 与"首跃"位置描述（不另立术语，只统计形状）
    gain = g[:, 3] - g[:, 0]
    shape = np.select([g[:, 3] < 0.05, gain < 0.02, (g[:, 1] - g[:, 0]) >= 0.8 * gain, (g[:, 2] - g[:, 0]) >= 0.8 * gain],
                      ["M3<0.05（弱且无增强）", "有推断能力但无背景增强", "增强主要在 K=1 完成", "增强主要在 K=2 完成"], "增强延续到 K=3")
    P["shape"] = shape
    sh = P.groupby("shape").agg(条数=("M0", "size"), M0均值=("M0", "mean"), M1均值=("M1", "mean"), M2均值=("M2", "mean"), M3均值=("M3", "mean"))
    rep.append(f"## {ds} 升级曲线形状统计\n\n" + md(sh))
    top = P.assign(gain1=P.M1 - P.M0).sort_values("gain1", ascending=False).head(12)
    rep.append(f"## {ds} K=1 增强最大的 12 条（见证背景）\n\n" + top[["conf", "field", "M0", "M1", "M2", "M3", "W1"]].to_markdown(index=False, floatfmt=".3f"))
    # ---------------- T5 基底场景
    if ds == "pjm":
        bm = pickle.load(open(ROOT / "outputs/sets/pjm_base_meta.pkl", "rb")); kb = bm["keys"]
        Tb = load_truth("pjm", "base", 0, len(kb))
        if Tb is not None:
            Vbb = closure(Tb["clean"], kb, Vval=Tb["val_r2"], has_empty=True)
            actb = [a for a in act if a not in bm["base"]]
            Mb, Wb = mk_table(Vbb, kb, actb, K_list=(0, 1, 2), base=True)
            VB = Vbb[index_of(kb)[()]]
            rows = []
            for c in range(12):
                pb = [act.index(a) for a in actb]
                rows.append(dict(conf=conf[c], V_B=VB[c], 基底泄露_tau07=bool(VB[c] > 0.7),
                                 最大M0_B空=M4[pb, 0, c].max(), 最大M0_给定B=Mb[:, 0, c].max(),
                                 最大M2_B空=M4[pb, 2, c].max(), 最大M2_给定B=Mb[:, 2, c].max(),
                                 M2排名Kendall_B空vs给定B=kendalltau(M4[pb, 2, c], Mb[:, 2, c])[0]))
            rep.append("## pjm T5 公开基底场景（B = 最新 + 日前负荷预测）\n\n" + md(pd.DataFrame(rows).set_index("conf")))
            np.savez(A / "pjm_M_base.npz", M=Mb, W=Wb, active=actb, VB=VB)
    rep.append(f"## {ds} 其他\n\n```\n{json.dumps(res, ensure_ascii=False, indent=1)}\n```")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["pjm", "caiso"]:
        run(ds)
    (A / "report_truth.md").write_text("\n\n".join(rep), encoding="utf-8"); print("\n\n".join(rep))
