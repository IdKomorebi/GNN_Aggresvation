# -*- coding: utf-8 -*-
"""127 号 步骤 1：全模型（3 种子）训练 → Dropout（全部规模 ≤3 集合，3 种子平均）→ 热启动+早停（全部集合，种子 0）。
集合顺序 = 122 号 eval_masks 的前 n3 行（与各号 keys 中规模 ≤3 的顺序一致）。用法：run_full.py --gpu 1 --tags ..."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import base127 as B  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--tags", default="RTS-GMLC,NEM,PJM-load,PJM-gen/ic,CAISO-load")
ap.add_argument("--skip_ws", action="store_true"); a = ap.parse_args(); dev = f"cuda:{a.gpu}"
for tag in a.tags.split(","):
    T = tag.replace("/", "_"); rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; D = registry.load_ds(rel)["D"]
    n3 = json.load(open(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_sets.json")))["n3"]
    M = np.load(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_masks.npy"))[:n3]
    O = os.path.join(ROOT, "outputs", "est", T); os.makedirs(O, exist_ok=True); os.makedirs(os.path.join(ROOT, "outputs", "full"), exist_ok=True)
    models = []
    for sd in range(3):
        f = os.path.join(ROOT, "outputs", "full", f"{T}_seed{sd}.pt")
        if os.path.exists(f):
            m = B.MLP(D["Xtr"].shape[1], D["Ytr"].shape[1]).to(dev); m.load_state_dict(torch.load(f, map_location=dev)); models.append(m.eval())
        else:
            m, info = B.train_full(D, sd, dev); torch.save(m.state_dict(), f); models.append(m); print(tag, "全模型", sd, info, flush=True)
    if not os.path.exists(os.path.join(O, "dropout.npz")):
        t0 = time.time(); e = B.dropout_est(D, models, M, dev); np.savez(os.path.join(O, "dropout.npz"), est=e, sec=(time.time() - t0) / n3)
        print(tag, "Dropout 完成", flush=True)
    if not a.skip_ws and not os.path.exists(os.path.join(O, "ws.npz")):
        t0 = time.time(); e = B.ws_est(D, models[0], M, dev); np.savez(os.path.join(O, "ws.npz"), est=e, sec=(time.time() - t0) / n3)
        print(f"[{time.strftime('%H:%M:%S')}] {tag} 热启动 完成 {time.time() - t0:.0f}s", flush=True)
print("全部完成")
