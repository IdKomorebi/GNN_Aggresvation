# -*- coding: utf-8 -*-
"""训练改造后的 GRPO 定级策略。用法：CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py --seed 0"""
import os, sys, time, json, argparse, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(HERE, "src")); import env, grpo  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--B", type=int, default=8); ap.add_argument("--G", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta", type=float, default=0.02); ap.add_argument("--ent", type=float, default=0.01); ap.add_argument("--sync", type=int, default=10)
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(1234 + a.seed); torch.set_num_threads(1)
dev = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = ["RTS-GMLC", "PJM-load", "PJM-gen/ic"]; TAUS = (0.5, 0.6, 0.7)
OUT = os.path.join(HERE, "outputs", "grpo"); os.makedirs(OUT, exist_ok=True)
BASE = {(t, tau): [env.minimal_sets(env.load_group(t)["V"], env.load_group(t)["keys"], c, tau) for c in range(env.load_group(t)["C"])] for t in TRAIN for tau in TAUS}


def new_instance(r):
    while True:
        t = TRAIN[r.integers(len(TRAIN))]; tau = TAUS[r.integers(len(TAUS))]
        ins = env.Instance(env.load_group(t), tau, r, viol_sets=BASE[(t, tau)], **env.sample_cfg(r))
        if len(ins.gl) > 0:
            return ins


VAL = [new_instance(np.random.default_rng(5_000_000 + k)) for k in range(36)]
VAL_REF = np.mean([v.cost(env.prune(v, env.greedy_cover(v))) / max(1e-9, v.cost(env.optimal(v))) for v in VAL])
pol = grpo.Policy().to(dev); ref = copy.deepcopy(pol); opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
log = []; best = np.inf; t0 = time.time()
print(f"设备 {dev}；验证集上 greedy_cover_prune 与最优之比 = {VAL_REF:.4f}", flush=True)
for it in range(a.iters + 1):
    if it % 20 == 0:   # 验证（只用训练组的验证实例）
        pol.eval()
        raw = np.mean([v.cost(grpo.rollout(v, pol, dev, greedy=True)[0]) / max(1e-9, v.cost(env.optimal(v))) for v in VAL])
        pr = np.mean([v.cost(env.prune(v, grpo.rollout(v, pol, dev, greedy=True)[0])) / max(1e-9, v.cost(env.optimal(v))) for v in VAL])
        log.append(dict(iter=it, val_raw=float(raw), val_prune=float(pr), sec=time.time() - t0))
        print(f"iter {it:4d}  验证：与最优之比 原始 {raw:.4f}  回退后 {pr:.4f}  （贪心基线 {VAL_REF:.4f}）  {time.time()-t0:.0f}s", flush=True)
        if pr < best - 1e-9:
            best = pr; torch.save(pol.state_dict(), os.path.join(OUT, f"policy_seed{a.seed}.pt"))
        json.dump(dict(args=vars(a), val_greedy=float(VAL_REF), log=log), open(os.path.join(OUT, f"trainlog_seed{a.seed}.json"), "w"), ensure_ascii=False, indent=1)
        pol.train()
    if it == a.iters:
        break
    if it % a.sync == 0:
        ref.load_state_dict(pol.state_dict())
    loss = 0.0
    for _ in range(a.B):
        ins = new_instance(rng); norm = max(1e-9, ins.cost(env.all_raise(ins)))
        trajs, Rs = [], []
        for _ in range(a.G):
            l, tr = grpo.rollout(ins, pol, dev, rng=rng, keep=True); trajs.append(tr); Rs.append(-ins.cost(l) / norm)
        Rs = np.array(Rs); adv = (Rs - Rs.mean()) / (Rs.std() + 1e-6)          # 组内相对优势
        for tr, ad in zip(trajs, adv):
            for ft, vt, act in tr:
                lp = pol(ft, vt); p = lp.exp()
                with torch.no_grad():
                    lr_ = ref(ft, vt)
                kl = (p[vt] * (lp[vt] - lr_[vt])).sum(); ent = -(p[vt] * lp[vt]).sum()
                loss = loss + (-float(ad) * lp[act] + a.beta * kl - a.ent * ent) / (a.B * a.G)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
print("完成；最优验证比值", best)
