# -*- coding: utf-8 -*-
"""86 号：复用 85 号缓存 poly2 充分统计量的 ms 级 v̂(S) 查询器。

不重训、不重造字典：直接加载 85 号 `outputs/poly2_moments.npz`，
对任意集合 key 抽取 raw+square+集合内两两积列，闭式解 ridge → 逐-conf test R²。
大集合可传 selected_edges 只保留稀疏 Top-K 乘积列（85 号做法）以控维度。
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
R85 = REPO / "DNN_Aggresvation85"
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402

ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])


class Poly2Query:
    def __init__(self, selected_edges=None):
        cache_path = R85 / "outputs/poly2_moments.npz"
        with np.load(cache_path) as payload:
            self.cache = {k: payload[k] for k in payload.files}
        cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
        cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
        data = prepare_data(cfg)
        self.n_general = int(data["n_general"])
        self.conf_names = list(data["confidential"])
        pairs = list(combinations(range(self.n_general), 2))
        self.pair_position = {p: i for i, p in enumerate(pairs)}
        self.selected_edges = selected_edges  # None=全两两积；set=稀疏

    def _columns(self, key: tuple[int, ...]) -> list[int]:
        ng = self.n_general
        cols = list(key) + [ng + i for i in key]
        for pair in combinations(key, 2):
            sp = tuple(sorted(pair))
            if self.selected_edges is None or sp in self.selected_edges:
                cols.append(2 * ng + self.pair_position[sp])
        return cols

    def _query(self, columns: list[int]) -> np.ndarray:
        c = self.cache
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
        gt, ht = c["g_test"][ix], c["h_test"][columns]
        sse = np.zeros(ha.shape[1])
        for ai, a in enumerate(ALPHAS):
            conf = np.flatnonzero(best == ai)
            if not len(conf):
                continue
            beta = np.linalg.solve(ga + a * eye, ha[:, conf])
            sse[conf] = c["rtr_test"][conf] - 2.0 * np.sum(beta * ht[:, conf], axis=0) + np.sum(beta * (gt @ beta), axis=0)
        return np.clip(1.0 - sse / c["target_total"], 0.0, None)

    @lru_cache(maxsize=3_000_000)
    def vhat(self, key: tuple[int, ...]) -> tuple[float, ...]:
        """逐-conf test R²（元组便于 lru_cache）。key 必须已排序去重。"""
        return tuple(self._query(self._columns(tuple(sorted(key)))))

    def vmax(self, key: tuple[int, ...]) -> float:
        return max(self.vhat(tuple(sorted(key))))

    def syn(self, key: tuple[int, ...]) -> tuple[float, int]:
        """纯高阶增量 max_c[ v_c(S) − max_{T⊂S,|T|=|S|-1} v_c(T) ]，返回 (syn, 最强conf索引)。"""
        key = tuple(sorted(key))
        v = np.array(self.vhat(key))
        subs = [tuple(x for x in key if x != drop) for drop in key]  # (m-1)-子集
        best_sub = np.max([np.array(self.vhat(s)) for s in subs], axis=0)
        inc = v - best_sub
        c = int(np.argmax(inc))
        return float(inc[c]), c
