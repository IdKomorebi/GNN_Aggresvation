# -*- coding: utf-8 -*-
"""88 号：三分割版查询器（修正"测试集参与搜索"）。

mode="search"：beam/peel/阈值/宽度等一切自适应决策在此进行，可反复使用；
mode="audit" ：**只在最终报数时用一次**，不得参与任何搜索或选择。

其余数值路径与 85 号完全一致（同一 alpha 网格、同一闭式解），仅换评估分片。
"""
from __future__ import annotations

import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402

ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])


class Poly2QuerySplit:
    def __init__(self, mode: str = "search", selected_edges=None):
        assert mode in ("search", "audit")
        self.mode = mode
        with np.load(ROOT / "outputs/poly2_moments_split.npz") as p:
            self.cache = {k: p[k] for k in p.files}
        cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
        cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
        data = prepare_data(cfg)
        self.n_general = int(data["n_general"])
        self.conf_names = list(data["confidential"])
        self.pair_position = {p: i for i, p in enumerate(combinations(range(self.n_general), 2))}
        self.selected_edges = selected_edges
        self.n_solve = 0

    def _columns(self, key):
        ng = self.n_general
        cols = list(key) + [ng + i for i in key]
        for pair in combinations(key, 2):
            sp = tuple(sorted(pair))
            if self.selected_edges is None or sp in self.selected_edges:
                cols.append(2 * ng + self.pair_position[sp])
        return cols

    def _query(self, columns):
        self.n_solve += 1
        c, m = self.cache, self.mode
        ix = np.ix_(columns, columns)
        gf, hf = c["g_fit"][ix], c["h_fit"][columns]
        gv, hv = c["g_val"][ix], c["h_val"][columns]
        eye = np.eye(len(columns))
        val = []
        for a in ALPHAS:
            beta = np.linalg.solve(gf + a * eye, hf)
            val.append(c["rtr_val"] - 2.0 * np.sum(beta * hv, axis=0) + np.sum(beta * (gv @ beta), axis=0))
        best = np.argmin(np.stack(val, axis=0), axis=0)
        ga, ha = c["g_all"][ix], c["h_all"][columns]
        ge, he = c[f"g_{m}"][ix], c[f"h_{m}"][columns]
        sse = np.zeros(ha.shape[1])
        for ai, a in enumerate(ALPHAS):
            conf = np.flatnonzero(best == ai)
            if not len(conf):
                continue
            beta = np.linalg.solve(ga + a * eye, ha[:, conf])
            sse[conf] = (c[f"rtr_{m}"][conf] - 2.0 * np.sum(beta * he[:, conf], axis=0)
                         + np.sum(beta * (ge @ beta), axis=0))
        return np.clip(1.0 - sse / c[f"total_{m}"], 0.0, None)

    @lru_cache(maxsize=3_000_000)
    def vhat(self, key):
        return tuple(self._query(self._columns(tuple(sorted(key)))))

    def vmax(self, key):
        return max(self.vhat(tuple(sorted(key))))

    def syn(self, key):
        key = tuple(sorted(key))
        v = np.array(self.vhat(key))
        subs = [tuple(x for x in key if x != d) for d in key]
        best = np.max([np.array(self.vhat(s)) for s in subs], axis=0)
        inc = v - best
        c = int(np.argmax(inc))
        return float(inc[c]), c
