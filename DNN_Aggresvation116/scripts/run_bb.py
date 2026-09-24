# -*- coding: utf-8 -*-
"""116 号：主干训练与估计的分阶段入口。
  --stage group_train --group G          本组主干 3 种子（pipe.train_oracle）→ groups/G/outputs/oracle_seed{s}.pt
  --stage full_train  --ds pjm|caiso     全列主干 3 种子（掩码重建）      → outputs/backbones/{ds}_full_seed{s}.pt
  --stage lto_train   --group G          留目标主干 3 种子（删去本组目标列）→ outputs/backbones/{G}_lto_seed{s}.pt
  --stage estimate    --group G --bb group|full|lto   新主口径 E，规模 ≤3 的集合 → groups/G/outputs/est_{bb}.npz
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import bb116  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation100", "src")); from common100 import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True); ap.add_argument("--group", default=""); ap.add_argument("--ds", default="")
ap.add_argument("--bb", default="group"); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args(); dev = f"cuda:{a.gpu}"
BB = os.path.join(ROOT, "outputs", "backbones"); os.makedirs(BB, exist_ok=True)
DS_OF = {"pjm_load": "pjm", "pjm_gen_ic": "pjm", "caiso_load": "caiso"}
log = open(os.path.join(ROOT, "logs", f"bb_{a.stage}_{a.group or a.ds}_{a.bb if a.stage == 'estimate' else ''}.log"), "a")


def say(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {a.stage} {a.group or a.ds} {a.bb if a.stage == 'estimate' else ''}: {msg}"
    print(line, flush=True); log.write(line + "\n"); log.flush()


def full_matrix(ds, drop=()):
    """数据集全部 53 列（去别名后的一般列 + 机密列），训练/测试行与各组 D.npz 完全一致。"""
    D0 = load(ds); gen, conf = D0["general"], D0["conf"]
    names = [gen[i] for i in D0["active"]] + conf
    Xtr = np.concatenate([D0["Xtr"][:, D0["active"]], D0["Ytr"]], 1).astype(np.float32)
    Xte = np.concatenate([D0["Xte"][:, D0["active"]], D0["Yte"]], 1).astype(np.float32)
    keep = [k for k, c in enumerate(names) if c not in drop]
    return Xtr[:, keep], Xte[:, keep], [names[k] for k in keep], D0["fit_idx"], D0["val_idx"]


def group_io(g):
    O = os.path.join(ROOT, "groups", g, "outputs")
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
    return O, spec, z


t0 = time.time()
if a.stage == "group_train":
    O, spec, z = group_io(a.group); D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
    for s in range(3):
        m, inf = pipe.train_oracle(D, s, dev); torch.save(m.state_dict(), os.path.join(O, f"oracle_seed{s}.pt")); say(f"种子 {s}：{inf}")
elif a.stage in ("full_train", "lto_train"):
    if a.stage == "full_train":
        ds, drop, tag = a.ds, (), f"{a.ds}_full"
    else:
        O, spec, z = group_io(a.group); ds = DS_OF[a.group]; drop = [t.replace("Y_", "") for t in spec["targ"]]; tag = f"{a.group}_lto"
    Xtr, Xte, names, fi, vi = full_matrix(ds, drop)
    json.dump(dict(columns=names, dropped=list(drop)), open(os.path.join(BB, f"{tag}_columns.json"), "w"), ensure_ascii=False, indent=1)
    for s in range(3):
        m, inf = bb116.train_mae(Xtr, fi, vi, s, dev); torch.save(m.state_dict(), os.path.join(BB, f"{tag}_seed{s}.pt")); say(f"种子 {s}：{inf}")
elif a.stage == "estimate":
    O, spec, z = group_io(a.group); ds = DS_OF[a.group]
    keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; sel = np.array([len(k) <= 3 for k in keys])
    masks = z["masks"][sel]; cand = spec["cand"]
    if a.bb == "group":
        models = bb116.load_models([os.path.join(O, f"oracle_seed{s}.pt") for s in range(3)], len(cand), z["Ytr"].shape[1], dev)
        Xtr, Xte, cmap = z["Xtr"], z["Xte"], list(range(len(cand)))
    else:
        tag = f"{ds}_full" if a.bb == "full" else f"{a.group}_lto"
        cols = json.load(open(os.path.join(BB, f"{tag}_columns.json"), encoding="utf-8"))["columns"]
        drop = [t.replace("Y_", "") for t in spec["targ"]] if a.bb == "lto" else ()
        Xtr, Xte, names, _, _ = full_matrix(ds, drop); assert names == cols
        cmap = [names.index(c) for c in cand]
        assert np.allclose(Xtr[:, cmap], z["Xtr"]), "行或列与本组 D.npz 不一致"
        models = bb116.load_models([os.path.join(BB, f"{tag}_seed{s}.pt") for s in range(3)], len(names), len(names), dev)
    say(f"开始：{len(masks)} 个集合，主干输入 {Xtr.shape[1]} 列")
    E, E1, sec = bb116.estimate_E(Xtr, z["Ytr"], Xte, z["Yte"], models, masks, cmap, dev, log=say)
    np.savez(os.path.join(O, f"est_{a.bb}.npz"), sel=sel, E=E, E1=E1, sec_per_set=sec)
    say(f"估计完成：{sec*1000:.0f} ms/集合")
say(f"完成，用时 {time.time()-t0:.0f}s")
