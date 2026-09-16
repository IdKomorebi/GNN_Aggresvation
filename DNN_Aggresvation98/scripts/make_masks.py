# -*- coding: utf-8 -*-
"""生成 98 号全部掩码集（44 维，列序 = 69 号 general 顺序）。

gameA : 14 玩家全部 2^14 联盟（精确 Shapley/交互基准，含已知协同/近重复/精确重复）
gameB : 12 个随机不同字段全部 2^12 联盟（无人工挑选的对照博弈）
rand  : 44 维上的保真度评估集 —— 单字段 44 + 字段对 946 + 规模均匀随机 1500
        + 边际配对 (S, S+i) ×400 + 交互四元组 (S,S+i,S+j,S+ij) ×200
"""
import sys, json
from itertools import combinations
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_pjm, GAME_A, game_b, game_masks

D = load_pjm(); g = D["general"]; nG = len(g)
out = ROOT / "outputs/masks"; out.mkdir(parents=True, exist_ok=True)
A = [g.index(f) for f in GAME_A]; Bf = game_b(g); B = [g.index(f) for f in Bf]
np.save(out / "gameA.npy", game_masks(A, nG)); np.save(out / "gameB.npy", game_masks(B, nG))

rng = np.random.RandomState(9801)
rows, meta = [], []
def add(idx, kind, **kw):
    m = np.zeros(nG, np.float32); m[list(idx)] = 1; rows.append(m); meta.append(dict(kind=kind, size=int(m.sum()), **kw))
for i in range(nG): add([i], "single", i=i)
for i, j in combinations(range(nG), 2): add([i, j], "pair", i=i, j=j)
for _ in range(1500):
    k = rng.randint(1, nG + 1); add(rng.choice(nG, k, replace=False), "random")
for q in range(400):
    k = rng.randint(0, nG); S = rng.choice(nG, k, replace=False); i = rng.choice(np.setdiff1d(np.arange(nG), S))
    add(S, "marg_S", q=q, i=int(i)); add(list(S) + [i], "marg_Si", q=q, i=int(i))
for q in range(200):
    k = rng.randint(0, nG - 1); S = rng.choice(nG, k, replace=False)
    i, j = rng.choice(np.setdiff1d(np.arange(nG), S), 2, replace=False)
    for tag, idx in [("S", S), ("Si", list(S) + [i]), ("Sj", list(S) + [j]), ("Sij", list(S) + [i, j])]:
        add(idx, "quad_" + tag, q=q, i=int(i), j=int(j))
np.save(out / "rand.npy", np.stack(rows))
json.dump(dict(gameA=GAME_A, gameB=Bf, rand_meta=meta, general=g, conf=D["conf"]), open(out / "masks_meta.json", "w"), ensure_ascii=False)
print("gameA", 2 ** len(A), "gameB", 2 ** len(B), "rand", len(rows)); print("gameB players:", Bf)
