"""把图2的两个面板左右拼接并加图注。"""
# 先渲染：dot -Tpng -Gdpi=150 图3a_两两关联图.dot -o _a.png；图3b_组合推断.dot -o _b.png
from PIL import Image, ImageDraw, ImageFont
a, b = Image.open("_a.png"), Image.open("_b.png")
F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
cap = [
    "图3  同一推断机制的两种表示（RTS-GMLC 受控市场算例：机组 313_CC_1 的私有报价加成 κ，逐小时 ±5%，11 个字段全部 2,047 个子集精确重训）",
    "左：两两关联图只看“一个字段—目标”，电价与目标的相关性和单字段推断能力都≈0；右：机组出力与任一电价合起来才能反推报价，",
    "增益以“本节点电价”最大、线路阻塞后的远端电价最小，与调度物理一致。电价是法定公开信息（进公开基底 B），需要管住的是“最后一块拼图”机组出力。",
]
font = ImageFont.truetype(F, 26)
gap, pad, lh = 40, 30, 40
W = a.width + b.width + gap + 2 * pad
H = max(a.height, b.height) + 2 * pad + lh * len(cap) + 20
img = Image.new("RGB", (W, H), "#fcfcfb")
img.paste(a, (pad, pad)); img.paste(b, (pad + a.width + gap, pad))
d = ImageDraw.Draw(img)
y = pad + max(a.height, b.height) + 20
for i, line in enumerate(cap):
    d.text((pad, y + i * lh), line, fill="#0b0b0b" if i == 0 else "#52514e", font=font)
img.save("../图3_组合推断示例.png")
print(img.size)
