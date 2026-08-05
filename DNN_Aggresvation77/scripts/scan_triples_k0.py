# -*- coding: utf-8 -*-
"""H-1 第一级：用 K=0 oracle 对**全部 13244 个三元组**做免费估计。

为什么能"免费"
--------------
K=0 不需要任何微调，所以**模型只构建一次**，之后就是纯前向。把多个掩码堆成一批
（每批 64 个掩码 × 1967 条测试样本）能把 13244 次查询压到分钟级。
对照：68 号原版是 K=200 全扫（每个三元组都要重建模型 + 200 步微调），约 90 分钟。

红线
----
剪枝器只用 oracle 估计。二元/单元的 v̂ 同样由本 oracle 在 K=0 给出
（不查任何重训真值），因此
    ŝyn3 = v̂(ijk) − max(v̂(ij), v̂(ik), v̂(jk))
整条链零真值在环。

输出：outputs/triples_k0.parquet（或 csv）——列 i,j,k,conf,v_ijk
      outputs/lowfour_k0.csv —— 990 个低阶子集的 K=0 v̂（供算 ŝyn3）
"""
import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

R77 = Path(__file__).resolve().parents[1]
R75 = R77.parent / "DNN_Aggresvation75"
R69 = R77.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R77 / "src"))

from src.data_processing import prepare_data   # noqa: E402
from src.oracle import MLPOracle               # noqa: E402
from runlog import log                          # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
    gi, ci = np.array(di["general_indices"]), np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    CONF = di["confidential"]
    xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
    yte = torch.as_tensor(di["test_data"][:, ci], dtype=torch.float32, device=DEV)
    ymean = yte.mean(0, keepdim=True)
    sst = ((yte - ymean) ** 2).sum(0) + 1e-12          # (nC,)

    ck = torch.load(R75 / "outputs" / f"oracle_{args.scheme}_seed{args.seed}.pt",
                    map_location=DEV, weights_only=False)
    model = MLPOracle(nG, nC)
    model.load_state_dict(ck["state"])
    model = model.to(DEV).eval()

    N = len(xte)

    @torch.no_grad()
    def batch_r2(masks: np.ndarray) -> np.ndarray:
        """masks: (B, nG) -> (B, nC) 的逐 conf R²。一次前向算 B 个掩码。"""
        B = len(masks)
        m = torch.as_tensor(masks, dtype=torch.float32, device=DEV)      # (B,nG)
        x = xte.unsqueeze(0).expand(B, -1, -1).reshape(B * N, nG)        # (B*N,nG)
        mm = m.unsqueeze(1).expand(-1, N, -1).reshape(B * N, nG)
        pred = model(x, mm).reshape(B, N, nC)
        ssr = ((yte.unsqueeze(0) - pred) ** 2).sum(1)                    # (B,nC)
        return torch.clamp(1 - ssr / sst.unsqueeze(0), min=0).cpu().numpy()

    # ---------- 低阶（44 单 + 946 对）----------
    low_keys = [(i,) for i in range(nG)] + list(combinations(range(nG), 2))
    rows = []
    for s in range(0, len(low_keys), MASK_BATCH):
        chunk = low_keys[s:s + MASK_BATCH]
        masks = np.zeros((len(chunk), nG), dtype=np.float32)
        for a, key in enumerate(chunk):
            masks[a, list(key)] = 1.0
        r2 = batch_r2(masks)
        for a, key in enumerate(chunk):
            for j, c in enumerate(CONF):
                rows.append(dict(key="_".join(f"{x:02d}" for x in key),
                                 size=len(key), conf=c, est=float(r2[a, j])))
    pd.DataFrame(rows).to_csv(R77 / "outputs/lowfour_k0.csv", index=False)
    print(f"低阶 {len(low_keys)} 个子集完成 [{time.time()-t0:.0f}s]", flush=True)

    # ---------- 三阶全扫 ----------
    tri = list(combinations(range(nG), 3))
    rows = []
    for s in range(0, len(tri), MASK_BATCH):
        chunk = tri[s:s + MASK_BATCH]
        masks = np.zeros((len(chunk), nG), dtype=np.float32)
        for a, key in enumerate(chunk):
            masks[a, list(key)] = 1.0
        r2 = batch_r2(masks)
        for a, (i, j, k) in enumerate(chunk):
            for q, c in enumerate(CONF):
                rows.append(dict(i=i, j=j, k=k, conf=c, est=float(r2[a, q])))
        if (s // MASK_BATCH) % 50 == 0:
            print(f"  三阶 {s}/{len(tri)} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R77 / "outputs/triples_k0.csv", index=False)

    el = time.time() - t0
    print(f"完成：{len(tri)} 个三元组 × {nC} conf，K=0 全扫 [{el:.0f}s]", flush=True)
    log("H-1", "DONE", note=f"K=0 全扫 {len(tri)} 三元组 + {len(low_keys)} 低阶子集"
                            f"（{args.scheme} seed{args.seed}）", elapsed_s=el)


if __name__ == "__main__":
    main()
