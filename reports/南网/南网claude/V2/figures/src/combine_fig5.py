"""图5：耦合回路（左）+ 示例表（右）。先渲染：dot -Tpng -Gdpi=140 图5a_耦合回路.dot -o _a.png；图5b_示例.dot -o _b.png"""
from PIL import Image, ImageDraw, ImageFont
a, b = Image.open("_a.png"), Image.open("_b.png")
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25)
cap = "图5  等级—可见范围—背景—风险的耦合，以及“全部上调”与“一致性优化”的差别"
pad, gap = 30, 30
W = pad * 2 + a.width + gap + b.width; H = pad * 2 + max(a.height, b.height) + 55
img = Image.new("RGB", (W, H), "#fcfcfb")
img.paste(a, (pad, pad + (max(a.height, b.height) - a.height) // 2)); img.paste(b, (pad + a.width + gap, pad + (max(a.height, b.height) - b.height) // 2))
ImageDraw.Draw(img).text((pad, H - 60), cap, fill="#0b0b0b", font=font)
img.save("../图5_等级一致性与示例.png"); print(img.size)
