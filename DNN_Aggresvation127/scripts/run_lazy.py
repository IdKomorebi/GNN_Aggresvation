# -*- coding: utf-8 -*-
"""127 号 步骤 2：LazyVI（双精度，忠实实现）只在每组同一批 200 个集合上运行（与 126 号计时同一随机种子 2 抽取）。
全表需约 44 GPU 小时（8.8 s/集合），不可行。输出 outputs/est/<组>/lazy_s200.npz（est、idx、sec）。依赖 run_full 的全模型种子 0。"""
import os, sys, json, time, argparse
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import base127 as B  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--tags", required=True); a = ap.parse_args(); dev = f"cuda:{a.gpu}"
for tag in a.tags.split(","):
    T = tag.replace("/", "_"); rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; D = registry.load_ds(rel)["D"]
    n3 = json.load(open(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_sets.json")))["n3"]
    M = np.load(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_masks.npy"))[:n3]
    idx = np.random.RandomState(2).choice(n3, min(200, n3), replace=False)
    f = os.path.join(ROOT, "outputs", "full", f"{T}_seed0.pt")
    while not os.path.exists(f):
        time.sleep(30)
    m = B.MLP(D["Xtr"].shape[1], D["Ytr"].shape[1]).to(dev); m.load_state_dict(torch.load(f, map_location=dev)); m.eval()
    torch.cuda.synchronize(dev); t0 = time.time(); e = B.lazy_est(D, m, M[idx], dev, dtype=torch.float64); torch.cuda.synchronize(dev)
    np.savez(os.path.join(ROOT, "outputs", "est", T, "lazy_s200.npz"), est=e, idx=idx, sec=(time.time() - t0) / len(idx))
    print(f"[{time.strftime('%H:%M:%S')}] {tag} LazyVI 完成 {(time.time() - t0) / len(idx):.2f} s/集合", flush=True)
