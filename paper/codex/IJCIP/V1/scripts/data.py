"""Read-only timing aggregation from the existing experiment outputs."""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
REPO = next(p for p in Path(__file__).resolve().parents if (p / "DNN_Aggresvation125").is_dir())
R = lambda *p: str(REPO.joinpath(*p))
def _baseline_times():
    """每集合耗时（秒），五组的 [最小, 最大]：126 号（本文、线性、二次、代理模型）、127 号（Dropout、热启动、LazyVI、TabPFN）、118 号（串行重训）。"""
    import glob
    t126 = pd.concat([pd.read_csv(f) for f in glob.glob(R("DNN_Aggresvation126/outputs/time_fast_*.csv"))])
    t127 = pd.concat([pd.read_csv(f) for f in glob.glob(R("DNN_Aggresvation127/outputs/time_[A-Z]*.csv"))] + [pd.read_csv(R("DNN_Aggresvation127/outputs/time_tabpfn.csv"))])
    T = {}
    for k, src, key in [("ours", t126, "C3"), ("lin", t126, "lin"), ("poly", t126, "poly"), ("surrogate", t126, "head3"),
                        ("dropout", t127, "dropout"), ("ws", t127, "ws"), ("lazyvi", t127, "lazyvi"), ("tabpfn", t127, "tabpfn")]:
        v = src[src.方法 == key].每集合ms / 1000; T[k] = (v.min(), v.max())
    seq = [np.load(R("DNN_Aggresvation118", "outputs", "est", t, "seq.npz"))["t"].sum(1).mean() for t in ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen_ic", "CAISO-load"]]
    T["retrain"] = (min(seq), max(seq)); return T
