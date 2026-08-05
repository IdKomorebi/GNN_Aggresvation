# -*- coding: utf-8 -*-
"""84 号 杠杆A 评估：给定 oracle checkpoint，在 79 号 2197 个设计三元组上算 K=0 的 S1/S2。

v̂_c(S) = 测试集按掩码 S 前向的逐-conf R²（K=0，纯前向，无微调）。
每个三元组：syn3_c = v_c(ijk) − max(v_c(ij),v_c(ik),v_c(jk))；S1 = max_c syn3_c；S2 = max_c v_c(ijk)。
合并 79 号真值 split（syn3_true/strong/split/weight）。

守红线：S1/S2 只来自 oracle 估计；真值仅用于最后打分，不进任何排序键。
"""
from __future__ import annotations

import argparse
import ast
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R79 = REPO / "DNN_Aggresvation79"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.oracle import MLPOracle  # noqa: E402
from train_mono import load_data, per_conf_r2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load_data()
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general, n_conf = int(data["n_general"]), int(data["n_confidential"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = torch.as_tensor(data["test_data"][:, general_idx], dtype=torch.float32, device=device)
    y_test = data["test_data"][:, conf_idx]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MLPOracle(n_general, n_conf).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    split = pd.read_csv(R79 / "outputs/truth_design_split.csv")
    triples = [tuple(ast.literal_eval(s)) for s in split["indices"]]

    # 收集所有需要的子集掩码（三元 + 三个二元 + 三个单元）
    need = set()
    for (i, j, k) in triples:
        need.add((i, j, k))
        for pr in combinations((i, j, k), 2):
            need.add(pr)
        for s in (i, j, k):
            need.add((s,))
    need = sorted(need, key=lambda t: (len(t), t))

    # 逐掩码前向算逐-conf R²，缓存
    vcache: dict[tuple, np.ndarray] = {}
    B = 4096
    with torch.no_grad():
        for start in range(0, len(need), B):
            chunk = need[start:start + B]
            for sub in chunk:
                m = torch.zeros(len(x_test), n_general, device=device)
                m[:, list(sub)] = 1.0
                pred = model(x_test, m).cpu().numpy()
                vcache[sub] = per_conf_r2(pred, y_test)  # (nC,)

    rows = []
    for (i, j, k) in triples:
        vijk = vcache[(i, j, k)]                                   # (nC,)
        pairs = np.stack([vcache[tuple(sorted(pr))] for pr in combinations((i, j, k), 2)])  # (3,nC)
        best_pair = pairs.max(axis=0)                             # (nC,)
        syn3_c = vijk - best_pair                                 # (nC,)
        c_star = int(np.argmax(syn3_c))
        rows.append(dict(
            indices=(i, j, k),
            S1=float(syn3_c.max()),          # max_c 纯三阶增量
            S2=float(vijk.max()),            # max_c 联合泄露(上界筛)
            vijk_at_cstar=float(vijk[c_star]),
            bestpair_at_cstar=float(best_pair[c_star]),
            conf_star=c_star,
        ))
    # split 的 indices 是字符串，用解析后的元组对齐
    split2 = split.copy()
    split2["indices"] = triples
    out = pd.DataFrame(rows).merge(
        split2[["indices", "syn3_true", "weight", "strong", "split"]],
        on="indices", how="left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[eval] {Path(args.ckpt).name} → {args.out}  n={len(out)} "
          f"S1<0={(out.S1<0).sum()}  strong={int(out.strong.sum())}", flush=True)


if __name__ == "__main__":
    main()
