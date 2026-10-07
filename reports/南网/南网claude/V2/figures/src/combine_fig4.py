"""图3：嵌套层级（左）+ 角色表与时点（右，上下叠放）+ 图注。"""
# 先渲染：dot -Tpng -Gdpi=150 图4a_信息层级.dot -o _p1.png；图4b_角色表.dot -o _p2.png；图4c_披露时点.dot -o _p3.png
from PIL import Image, ImageDraw, ImageFont
p1, p2, p3 = (Image.open(f"_{k}.png") for k in ("p1", "p2", "p3"))
F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font = ImageFont.truetype(F, 26)
cap = ["图4  南网文件已经规定了“谁在什么时候知道什么”，这正是推断风险评估里的公开基底 B 与背景池 H",
       "左：按《信息披露实施细则》5.1 的三层级再加内部数据，外层能看到内层的一切；右上：由此定义各类接收方的 B 与 H；",
       "右下：同一目标随披露时点推进，其敏感窗口会关闭，对应分级手册附录E-3“时效性”场景。"]
pad, gap, lh = 30, 40, 40
right_w = max(p2.width, p3.width)
W = pad * 2 + p1.width + gap + right_w
body_h = max(p1.height, p2.height + gap + p3.height)
H = pad * 2 + body_h + 20 + lh * len(cap)
img = Image.new("RGB", (W, H), "#fcfcfb")
img.paste(p1, (pad, pad))
x = pad + p1.width + gap
img.paste(p2, (x, pad)); img.paste(p3, (x, pad + p2.height + gap))
d = ImageDraw.Draw(img)
y = pad + body_h + 20
for i, line in enumerate(cap):
    d.text((pad, y + i * lh), line, fill="#0b0b0b" if i == 0 else "#52514e", font=font)
img.save("../图4_角色时点与公开基底.png"); print(img.size)
