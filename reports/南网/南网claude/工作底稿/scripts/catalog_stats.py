"""开放目录（第二批）统计与配图。

用法：/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python catalog_stats.py
输入：../开放目录第二批.csv（由原 xlsx 经 LibreOffice 转出，前两行为标题）
输出：../开放目录_统计.json、../../figures/图4_开放目录统计.png
"""
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FIG = BASE.parent / "figures"

rows = list(csv.reader(open(BASE / "开放目录第二批.csv", encoding="utf-8")))
hdr = rows[2]
data = [r for r in rows[3:] if r and r[0].strip().isdigit()]
col = {h: i for i, h in enumerate(hdr)}


def cnt(name):
    return collections.Counter(r[col[name]].strip() for r in data)


def domain(r):
    d = r[col["所属业务域"]].strip()
    return "跨域（含系统运行域）" if "、" in d else d


dom = collections.Counter(domain(r) for r in data)
uncond = [r for r in data if r[col["开放策略"]].strip() == "无条件开放"]
cond_kind = collections.Counter()
for r in data:
    c = r[col["开放条件"]]
    if r[col["开放策略"]].strip() == "无条件开放":
        cond_kind["无条件开放（公开途径下载或按披露细则披露）"] += 1
    elif "信息披露" in c:
        cond_kind["按电力市场信息披露规则向市场成员披露"] += 1
    elif "授权同意" in c:
        cond_kind["须获个人/企业/外部机构授权或脱敏后开放 + 签协议"] += 1
    else:
        cond_kind["签订数据使用协议 + 经可信数据空间开放"] += 1

stats = {
    "条目总数": len(data),
    "开放策略": dict(cnt("开放策略")),
    "业务域": dict(dom.most_common()),
    "开放对象": dict(cnt("开放对象").most_common()),
    "更新频率": dict(cnt("更新频率").most_common()),
    "数据区域范围": dict(cnt("数据区域范围").most_common()),
    "开放条件类型": dict(cond_kind.most_common()),
    "无条件开放条目": [f'{r[0]} {r[col["数据资源名称"]]}' for r in uncond],
    "引用2025版信息披露细则的条目数": sum("2025 年 V1.0 版" in r[col["开放条件"]] or "2025年V1.0版" in r[col["开放条件"]] for r in data),
    "是否有安全等级列": "否（列为：" + "、".join(hdr) + "）",
}
(BASE / "开放目录_统计.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 图：两个单序列横向条形图 ----------
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
INK, INK2, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6"

obj = collections.Counter(r[col["开放对象"]].strip() for r in data)
panels = [
    ("按业务域（561 项）", dom.most_common()),
    ("按开放对象（561 项）", obj.most_common()),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax, (title, items) in zip(axes, panels):
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    ax.set_facecolor(SURF)
    ax.barh(labels, vals, color=BLUE, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, y, str(v), va="center", fontsize=9, color=INK2)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.tick_params(axis="y", labelsize=9, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(0, max(vals) * 1.12)
fig.suptitle("南方电网公司数据开放目录（第二批）征求意见稿：548 项有条件开放、13 项无条件开放，均无安全等级列",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "图4_开放目录统计.png", dpi=170, facecolor=SURF)
print(json.dumps(stats, ensure_ascii=False, indent=1)[:1500])
