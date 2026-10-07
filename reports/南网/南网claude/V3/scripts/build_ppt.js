// V3：按组内项目汇报模板的风格（蓝色标题 + 细线 + 校徽、➢ 小标题、红色关键词、虚线要点框、分层框架图、编号步骤卡片）
// 运行：export PATH=/data1/duhaocun/envs/node_pptx/bin:$PATH NODE_PATH=/data1/duhaocun/envs/node_pptx_work/node_modules; node V3/scripts/build_ppt.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const V3 = path.resolve(__dirname, "..");
const A = (f) => path.join(V3, "assets", f);
const OUT = path.join(V3, "南网分类分级体系框架_V3.pptx");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
const FONT = "微软雅黑";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "南网电力数据分类分级体系框架";
// 模板配色
const TITLE = "005699", LINE = "3B4681", B1 = "5B9BD5", B0 = "DEEAF6", B2 = "1F4E79", NAVY = "002060", OR = "ED7D31", OR0 = "FBE5D6",
  YE = "FFC000", GR = "70AD47", GR0 = "E2F0D9", RED = "C00000", GRAY = "E7E6E6", HEAD = "4472C4", ARW = "9DC3E6", K = "000000", MUTE = "595959";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { line: { x: 0.76, y: 1.11, w: 11.81, h: 0, line: { color: LINE, width: 1 } } },
    { image: { path: A("logo_color.png"), x: 10.35, y: 0.4, w: 2.31, h: 0.62 } },
    { placeholder: { options: { name: "title", type: "title", x: 0.76, y: 0.36, w: 9.5, h: 0.72, fontFace: FONT, fontSize: 28, bold: true, color: TITLE, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.08, w: 0.6, h: 0.3, fontSize: 10, color: MUTE, align: "right" },
});
pres.defineSlideMaster({ title: "COVER", background: { path: A("cover_bg.png") }, objects: [{ image: { path: A("logo_white.png"), x: 5.54, y: 0.65, w: 2.25, h: 0.6 } }] });
pres.defineSlideMaster({ title: "OUTLINE", background: { color: "FFFFFF" }, objects: [{ image: { path: A("outline_header.png"), x: 0, y: 0, w: 13.33, h: 2.46 } }] });

// ---------------- 小工具 ----------------
let uid = 0; const nm = (p) => `${p}-${++uid}`;
// 标记：{{红色加粗}}  **加粗**  ^{上标}  _{下标}
function rt(s, base = {}) {
  const out = []; const re = /\{\{(.+?)\}\}|\*\*(.+?)\*\*|\^\{([^}]*)\}|_\{([^}]*)\}/g; let last = 0, m;
  const push = (t, o) => { if (t) out.push({ text: t, options: { ...base, ...o } }); };
  while ((m = re.exec(s))) {
    push(s.slice(last, m.index), {});
    if (m[1] !== undefined) push(m[1], { bold: true, color: RED });
    else if (m[2] !== undefined) push(m[2], { bold: true });
    else if (m[3] !== undefined) push(m[3], { superscript: true });
    else push(m[4], { subscript: true });
    last = re.lastIndex;
  }
  push(s.slice(last), {});
  return out;
}
// 把 run 里的 "\n" 拆成 breakLine，并给每个 run 带上对齐方式（否则换行后的段落会回到左对齐）
function norm(runs, align) {
  const out = [];
  runs.forEach((r) => {
    const parts = String(r.text).split("\n");
    parts.forEach((p, i) => { out.push({ text: p, options: { ...(r.options || {}), align, ...(i < parts.length - 1 ? { breakLine: true } : {}) } }); });
  });
  return out.filter((r, i) => r.text !== "" || r.options.breakLine);
}
function T(s, t, x, y, w, h, o = {}) {
  s.addText(typeof t === "string" ? rt(t) : t, { x, y, w, h, isTextBox: true, margin: 0, fontFace: FONT, fontSize: 16, color: K, valign: "top", align: "left", objectName: nm("text"), ...o });
}
function sub(s, t, y = 1.25) { T(s, [{ text: "➢ ", options: { bold: true } }, ...rt(t, { bold: true })], 0.76, y, 11.8, 0.45, { fontSize: 20, valign: "middle" }); }
function bul(s, items, x, y, w, h, o = {}) {
  const runs = [];
  items.forEach((it, i) => { const r = rt(it); r.forEach((q) => { q.options.bullet = { code: "25CF" }; }); r[r.length - 1].options.breakLine = i < items.length - 1; r.forEach((q) => { q.options.paraSpaceAfter = 5; }); runs.push(...r); });
  T(s, runs, x, y, w, h, { fontSize: 15, ...o });
}
function box(s, t, x, y, w, h, o = {}) {
  const { fill = B0, line = B1, color = K, fs = 12, bold = false, dash, lw = 1, align = "center", valign = "middle", round = false, margin = 0.04 } = o;
  s.addText(norm(typeof t === "string" ? rt(t) : t, align), {
    x, y, w, h, shape: round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, ...(round ? { rectRadius: 0.08 } : {}),
    fill: fill ? { color: fill } : { type: "none" }, line: line ? { color: line, width: lw, ...(dash ? { dashType: dash } : {}) } : { type: "none" },
    fontFace: FONT, fontSize: fs, bold, color, align, valign, margin, objectName: nm("box"),
  });
}
const dbox = (s, x, y, w, h, color = B1) => box(s, "", x, y, w, h, { fill: null, line: color, dash: "dash", lw: 1.25 });
function arrR(s, x, y, w = 0.45, h = 0.3, color = ARW) { s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h, fill: { color }, line: { color, width: 0 }, objectName: nm("arr") }); }
function arrD(s, x, y, w = 0.4, h = 0.3, color = ARW) { s.addShape(pres.shapes.DOWN_ARROW, { x, y, w, h, fill: { color }, line: { color, width: 0 }, objectName: nm("arr") }); }
function ln(s, x1, y1, x2, y2, o = {}) {
  const q = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color: o.color || "404040", width: o.width || 1.25, endArrowType: o.noArrow ? "none" : "triangle", ...(o.dash ? { dashType: o.dash } : {}) }, objectName: nm("ln") };
  if (x2 < x1) q.flipH = true; if (y2 < y1) q.flipV = true;
  s.addShape(pres.shapes.LINE, q);
}
// 编号步骤卡片（虚线框 + 彩色编号块）
function numCard(s, n, title, desc, x, y, w, h, color) {
  box(s, "", x, y, w, h, { fill: null, line: "404040", dash: "dash", lw: 1 });
  box(s, String(n), x + w / 2 - 0.5, y + 0.1, 1.0, 0.36, { fill: color, line: color, color: "FFFFFF", fs: 18, bold: true });
  T(s, title, x + 0.08, y + 0.52, w - 0.16, 0.34, { fontSize: 14, bold: true, align: "center", valign: "middle" });
  T(s, desc, x + 0.14, y + 0.9, w - 0.28, h - 1.0, { fontSize: 12 });
}
// 分层框架的一层：左侧深色竖排标签 + 右侧容器
function band(s, label, x, y, w, h, o = {}) {
  box(s, label.split("").join("\n"), x, y, 0.42, h, { fill: o.side || B2, line: o.side || B2, color: "FFFFFF", fs: 12, bold: true });
  box(s, "", x + 0.42, y, w - 0.42, h, { fill: o.fill || "FFFFFF", line: o.line || B1, lw: 1 });
}
const hd = (t, o = {}) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEAD }, fontFace: FONT, fontSize: 13, align: "center", valign: "middle", ...o } });
const td = (t, o = {}) => ({ text: typeof t === "string" ? rt(t) : t, options: { fontFace: FONT, fontSize: 12, color: K, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08], ...o } });
const BD = { type: "solid", pt: 0.75, color: "BFBFBF" };
const SRC = { nw: [B0, B1], sx: [OR0, OR], me: [GR0, GR] }; // 来源配色：南网 / 师兄 / 本研究
const sbox = (s, src, t, x, y, w, h, o = {}) => box(s, t, x, y, w, h, { fill: SRC[src][0], line: SRC[src][1], ...o });
function legend(s, x, y) {
  [["nw", "南网规则与流程"], ["sx", "师兄论文方法"], ["me", "本研究新增"]].forEach((l, i) => {
    box(s, "", x + i * 2.25, y + 0.04, 0.3, 0.2, { fill: SRC[l[0]][0], line: SRC[l[0]][1] });
    T(s, l[1], x + 0.38 + i * 2.25, y, 1.8, 0.28, { fontSize: 11, color: MUTE, valign: "middle" });
  });
}
function content(title) { const s = pres.addSlide({ masterName: "CONTENT" }); s.addText(title, { placeholder: "title" }); return s; }
function outline(active) {
  const s = pres.addSlide({ masterName: "OUTLINE" });
  T(s, "汇报提纲", 3.0, 0.75, 7.33, 1.0, { fontSize: 40, color: "FFFFFF", align: "center", valign: "middle" });
  ln(s, 3.9, 1.25, 4.9, 1.25, { color: "FFFFFF", noArrow: true, width: 1 }); ln(s, 8.43, 1.25, 9.43, 1.25, { color: "FFFFFF", noArrow: true, width: 1 });
  ["背景与问题", "体系框架", "关键方法", "下一步工作"].forEach((t, i) => {
    const y = 2.95 + i * 1.02, on = i === active;
    s.addText(String(i + 1), { x: 4.27, y, w: 0.66, h: 0.66, shape: pres.shapes.OVAL, fill: { color: on ? NAVY : "D9D9D9" }, line: { color: on ? NAVY : "D9D9D9", width: 0 }, fontFace: "Times New Roman", fontSize: 22, bold: true, italic: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: nm("num") });
    T(s, t, 5.27, y, 6, 0.66, { fontSize: 28, bold: true, color: on ? K : "BFBFBF", valign: "middle" });
  });
  return s;
}

// ================= 1 封面 =================
let s = pres.addSlide({ masterName: "COVER" });
T(s, "南网电力数据分类分级体系框架", 0.9, 1.45, 11.53, 1.0, { fontSize: 50, bold: true, color: YE, align: "center", valign: "middle" });
T(s, "融合关联推断风险的分类分级方法", 0.9, 2.5, 11.53, 0.8, { fontSize: 32, bold: true, color: YE, align: "center", valign: "middle" });
ln(s, 2.2, 3.72, 4.4, 3.72, { color: "FFFFFF", noArrow: true, width: 1 }); ln(s, 8.93, 3.72, 11.13, 3.72, { color: "FFFFFF", noArrow: true, width: 1 });
T(s, "数据分类分级专题汇报", 4.4, 3.52, 4.53, 0.4, { fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
T(s, "西安交通大学 网络空间学院", 3.4, 4.1, 6.53, 0.4, { fontSize: 18, bold: true, color: "FFFFFF", align: "center" });
T(s, "2026 年 10 月", 3.4, 4.55, 6.53, 0.4, { fontSize: 18, bold: true, color: "FFFFFF", align: "center" });

// ================= 2 提纲 =================
outline(0);

// ================= 3 背景：南网要求 =================
s = content("研究背景：南网数据分类分级要求");
dbox(s, 0.76, 1.3, 11.81, 1.0);
T(s, "南网 2026 年《数据分类分级工作手册》按{{影响对象}}和{{影响程度}}把数据定为 {{1~5 级}}，遵循{{“就高不就低”}}；分类一直细到数据字段，结果经审批后固化到数据目录并动态更新。", 0.95, 1.36, 11.45, 0.9, { fontSize: 17, bold: true, valign: "middle" });
T(s, "工作流程", 0.76, 2.5, 3, 0.35, { fontSize: 15, bold: true, color: TITLE });
["数据资产梳理", "数据分类", "数据分级", "审批", "结果固化", "技术管控", "动态更新"].forEach((t, i) => {
  s.addText(t, { x: 0.76 + i * 1.69, y: 2.9, w: 1.78, h: 0.55, shape: i === 0 ? pres.shapes.PENTAGON : pres.shapes.CHEVRON, fill: { color: i === 2 ? B2 : B1 }, line: { color: "FFFFFF", width: 1 }, fontFace: FONT, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: nm("chev") });
});
T(s, "分级判定方法", 0.76, 3.7, 3, 0.35, { fontSize: 15, bold: true, color: TITLE });
box(s, "**8 个分级要素**\n领域、群体、区域、重要性\n精度、规模、覆盖度、{{深度}}", 0.76, 4.1, 2.7, 1.25, { fs: 12 });
arrR(s, 3.52, 4.57);
box(s, "**影响对象**\n群体：国家安全、公共利益、\n经济运行、社会秩序\n个体：个人、其他组织\n单位自身", 4.03, 4.1, 2.7, 1.25, { fs: 11 });
T(s, "×", 6.75, 4.5, 0.35, 0.4, { fontSize: 20, bold: true, align: "center", valign: "middle" });
box(s, "**影响程度**\n无 / 轻微 / 一般 /\n严重 / 特别严重", 7.12, 4.1, 2.0, 1.25, { fs: 12 });
arrR(s, 9.18, 4.57);
[["1 级", "DEEAF6", K], ["2 级", "BDD7EE", K], ["3 级", "9DC3E6", K], ["4 级", "2E75B6", "FFFFFF"], ["5 级", B2, "FFFFFF"]].forEach((l, i) => box(s, l[0], 9.7 + i * 0.575, 4.1, 0.575, 0.6, { fill: l[1], line: "FFFFFF", color: l[2], fs: 12, bold: true }));
box(s, "一般数据", 9.7, 4.75, 1.725, 0.6, { fill: "F2F2F2", line: "BFBFBF", fs: 12 });
box(s, "重要", 11.425, 4.75, 0.575, 0.6, { fill: "F2F2F2", line: "BFBFBF", fs: 11 });
box(s, "核心", 12.0, 4.75, 0.575, 0.6, { fill: "F2F2F2", line: "BFBFBF", fs: 11 });
T(s, "各级的共享开放口径", 0.76, 5.6, 3, 0.35, { fontSize: 15, bold: true, color: TITLE });
s.addTable([
  [hd("等级"), hd("1 级"), hd("2 级"), hd("3 级"), hd("4、5 级")],
  [td("共享开放", { bold: true, align: "center" }), td("一般对象无条件共享、开放", { align: "center" }), td("有条件开放", { align: "center" }), td("仅特定对象有条件共享、开放", { align: "center" }), td("原则上不共享、不开放；须由主管部门认定", { align: "center" })],
], { x: 0.76, y: 6.0, w: 11.81, colW: [1.4, 2.9, 1.9, 2.9, 2.71], rowH: [0.36, 0.5], border: BD, objectName: nm("table") });
s.addNotes("南网2026年的分类分级工作手册是整个体系的基线。它规定了七步流程、八个分级要素，以及按影响对象和影响程度定一到五级的判定矩阵。每一级还绑定了共享开放的口径，这一点在后面的等级优化里会用到。");

// ================= 4 问题 =================
s = content("问题：数据关联带来的推断风险");
dbox(s, 0.76, 1.3, 11.81, 1.0);
T(s, "现行分级回答“{{这份数据本身泄露了有多严重}}”，没有回答“{{它和接收方已有的数据放在一起，能推出什么}}”：若干单看等级不高的字段，组合后可能还原出高等级数据。", 0.95, 1.36, 11.45, 0.9, { fontSize: 17, bold: true, valign: "middle" });
// 左：示意
T(s, "低等级字段的组合推断（示意）", 0.76, 2.5, 5.6, 0.35, { fontSize: 15, bold: true, color: TITLE, align: "center" });
[["字段 A：出清电价", "1 级 · 法定公开"], ["字段 B：机组日发电量", "1 级"], ["字段 C：新能源出力预测", "1 级"]].forEach((f, i) => {
  box(s, [{ text: f[0] + "\n", options: { bold: true, fontSize: 13 } }, { text: f[1], options: { fontSize: 11, color: MUTE } }], 0.76, 2.95 + i * 0.95, 2.35, 0.75, {});
  ln(s, 3.11, 3.325 + i * 0.95, 3.75, 4.27, { color: B1, width: 1.5 });
});
box(s, "关联\n推断", 3.75, 3.87, 0.9, 0.8, { fill: OR, line: OR, color: "FFFFFF", fs: 14, bold: true, round: true });
arrR(s, 4.72, 4.12, 0.5, 0.3, OR);
box(s, [{ text: "敏感目标：机组报价\n", options: { bold: true, fontSize: 12, color: "FFFFFF" } }, { text: "3 级 · 特定信息", options: { fontSize: 11, color: "FFFFFF" } }], 5.28, 3.87, 1.85, 0.8, { fill: RED, line: RED });
T(s, "单看：每个字段对目标的推断能力都很低\n合起来：推断精度可越过“视为泄露”的界线", 0.76, 5.75, 6.1, 0.7, { fontSize: 13, bold: true, color: RED, align: "center" });
// 右：南网条款
T(s, "南网文件已提出要求，但未给出方法与判定标准", 7.25, 2.5, 5.32, 0.35, { fontSize: 15, bold: true, color: TITLE, align: "center" });
s.addTable([
  [hd("出处"), hd("要求")],
  [td("分级原则“从严性”", { bold: true }), td("多因素时按可能造成的最高影响定级")],
  [td("分级要素“深度”", { bold: true }), td("关联、挖掘、融合对隐含信息的刻画程度")],
  [td("附录 D 衍生数据", { bold: true }), td("挖掘出更敏感、更深度的数据应提高级别")],
  [td("附录 E 聚合场景", { bold: true }), td("须分析汇聚后能否获得更多信息")],
  [td("风险评估第 382 项", { bold: true }), td("基于已公开数据，尝试能否推断出未公开的关联信息")],
], { x: 7.25, y: 2.95, w: 5.32, colW: [1.95, 3.37], rowH: 0.52, border: BD, objectName: nm("table") });
T(s, "本框架的目标：把这一原则性要求变成{{可计算、可复核}}的定级依据", 7.25, 6.2, 5.32, 0.6, { fontSize: 13, bold: true, align: "center" });
s.addNotes("现行分级看的是数据本身。但几个单看都是一级的字段，放在同一个接收方手里，可能把一份三级数据推出来。南网文件在五处提到了这类要求，但都没有给出怎么算、算到什么程度算有风险。");

// ================= 5 提纲 =================
outline(1);

// ================= 6 总体思路 =================
s = content("总体思路");
dbox(s, 0.76, 1.3, 11.81, 1.0);
T(s, "以{{南网分类分级规则为基线}}，继承师兄“规则初始分级—关联风险识别—动态定级”的体系框架，把推断风险由字段两两关联升级为{{集合与背景条件下的推断风险}}，形成符合南网规则的分类分级体系。", 0.95, 1.36, 11.45, 0.9, { fontSize: 17, bold: true, valign: "middle" });
T(s, "原体系（师兄论文）", 0.76, 2.5, 5.2, 0.35, { fontSize: 15, bold: true, color: TITLE, align: "center" });
T(s, "本框架", 7.37, 2.5, 5.2, 0.35, { fontSize: 15, bold: true, color: TITLE, align: "center" });
const rows6 = [
  ["规则理解", "大模型 + 检索增强抽取规则", "sx", "语料换为南网手册与细则", "sx", "保留"],
  ["初始定级", "频繁模式支持度 → 四级", "sx", "南网影响矩阵 → 五级", "nw", "按南网替换"],
  ["风险识别", "Pearson/Spearman/MIC 关联图\nDNN、GAT 推断验证", "sx", "关联图与推断验证保留\n+ 集合推断能力与字段级风险", "me", "升级"],
  ["等级决策", "GRPO 逐字段选等级\n合规为罚项", "sx", "GRPO 求上调最少的一致等级\n南网规则为硬约束", "sx", "保留并改造"],
  ["结果输出", "算法直接给出等级", "sx", "建议等级 + 推断证据 → 审批固化", "nw", "按南网对齐"],
];
rows6.forEach((r, i) => {
  const y = 2.95 + i * 0.78;
  box(s, r[0], 0.76, y, 1.2, 0.66, { fill: B2, line: B2, color: "FFFFFF", fs: 12, bold: true });
  sbox(s, r[2], r[1], 2.0, y, 3.96, 0.66, { fs: 12 });
  arrR(s, 6.05, y + 0.18, 0.5, 0.3);
  T(s, r[5], 5.75, y - 0.12, 1.1, 0.24, { fontSize: 10, color: RED, bold: true, align: "center" });
  sbox(s, r[4], r[3], 6.65, y, 5.92, 0.66, { fs: 12 });
});
legend(s, 0.76, 6.92);
s.addNotes("总体思路是三句话：南网规则是基线；师兄的体系框架保留；风险识别这一环升级。图中左边是原体系，右边是本框架，中间标出每个环节是保留、替换还是升级。");

// ================= 7 体系框架总图 =================
s = content("体系框架：四个阶段");
const BX = 0.76, BW = 11.81;
function row(yy, hh, label, items, opts = {}) {
  band(s, label, BX, yy, BW, hh);
  const n = items.length, gap = 0.42, x0 = BX + 0.6, wAll = BW - 0.42 - 0.36, w = (wAll - gap * (n - 1)) / n;
  items.forEach((it, i) => {
    const x = x0 + i * (w + gap);
    sbox(s, it[0], it[1], x, yy + 0.16, w, hh - 0.32, { fs: 11.5 });
    if (i < n - 1) arrR(s, x + w + 0.04, yy + hh / 2 - 0.13, gap - 0.08, 0.26);
  });
}
row(1.24, 1.12, "基础分级", [["sx", "**规则库构建**\n大模型 + 检索增强\n解析南网手册与细则"], ["nw", "**分类到字段**\n专业域—流程—\n业务对象—字段"], ["nw", "**规则定级**\n影响对象 × 影响程度\n→ 1~5 级"], ["nw", "**场景标注**\n披露层级、披露时点、\n是否法定公开"]]);
arrD(s, 6.45, 2.38, 0.44, 0.22);
row(2.62, 1.2, "风险识别", [["me", "**场景确定**\n各类接收方已知什么、\n还能拿到什么"], ["me", "**集合推断能力 V(S)**\n通用模型扫描字段组合\n（DNN/GAT 为其特例）"], ["me", "**字段级风险**\nM(0) 单独 · M(K) 最坏背景\nΓ(K) 背景放大"], ["me", "**重训验证**\n最强攻击模型、\n全量数据"]]);
sbox(s, "sx", "风险关联图（并联）：解释与先验", 4.32, 3.86, 3.6, 0.28, { fs: 10.5 });
arrD(s, 6.45, 4.16, 0.44, 0.22);
row(4.4, 1.15, "等级优化", [["me", "**一致性检查**\n任何角色都不能把高于其权限\n的目标推过泄露线"], ["sx", "**等级优化（GRPO）**\n上调某字段 → 可见范围收窄 → 风险重算\n目标：一致的前提下上调最少"], ["nw", "**建议等级 + 推断证据**\n写入“深度”要素与影响程度\n按矩阵复核"]]);
arrD(s, 6.45, 5.57, 0.44, 0.22);
row(5.81, 1.0, "审批复评", [["nw", "**审批**"], ["nw", "**固化到数据目录**"], ["nw", "**事件触发复评**（附录 E 场景、新批次开放、披露时点到达）"]]);
legend(s, 0.76, 6.95);
T(s, "硬约束：不低于规则等级 · 法定公开字段不调整 · 4、5 级只出申报建议", 7.0, 6.95, 5.57, 0.28, { fontSize: 11, bold: true, color: RED, align: "right", valign: "middle" });
s.addNotes("这是体系的总图，分四个阶段。第一和第四阶段是南网已有流程；第二阶段是推断风险识别，是本研究新增的部分，师兄的关联图并联使用；第三阶段用师兄的GRPO做等级优化，南网规则作为硬约束。");

// ================= 8 提纲 =================
outline(2);

// ================= 9 阶段一 =================
s = content("阶段一：按南网规则完成基础分类分级");
sub(s, "规则解析 → 分类 → 定级 → 场景标注");
bul(s, ["沿用师兄的{{大模型 + 检索增强}}方法，把南网手册、披露细则等解析为结构化规则库", "按业务架构线分类到字段，用{{影响对象 × 影响程度矩阵}}判定规则等级，替换原来的支持度定级", "同时标注每个字段的{{披露层级、披露时点、是否法定公开}}，供下一阶段使用"], 1.0, 1.75, 11.5, 1.2);
// 流程
[["分级手册", "披露细则", "开放目录"]].forEach(() => {});
box(s, "南网文件\n分级手册 · 风险手册\n披露细则 · 开放目录", 0.76, 3.3, 1.9, 1.1, { fill: GRAY, line: "A6A6A6", fs: 11 });
arrR(s, 2.72, 3.7);
sbox(s, "sx", "**大模型 + 检索增强**\n提示约束 · JSON 输出", 3.23, 3.3, 2.0, 1.1, { fs: 11 });
arrR(s, 5.29, 3.7);
sbox(s, "sx", "**结构化规则库**\n影响对象 · 影响程度\n依据条款 · 披露层级", 5.8, 3.3, 2.0, 1.1, { fs: 11 });
T(s, "线分类", 0.76, 4.65, 1.0, 0.3, { fontSize: 12, bold: true, color: TITLE });
["专业域", "流程组", "操作级流程", "业务对象", "数据字段"].forEach((t, i) => {
  s.addText(t, { x: 0.76 + i * 1.4, y: 5.0, w: 1.5, h: 0.5, shape: i === 0 ? pres.shapes.PENTAGON : pres.shapes.CHEVRON, fill: { color: i === 4 ? B2 : B1 }, line: { color: "FFFFFF", width: 1 }, fontFace: FONT, fontSize: 11, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: nm("chev") });
});
T(s, "每个字段的输出", 0.76, 5.75, 2.0, 0.3, { fontSize: 12, bold: true, color: TITLE });
["规则等级", "依据条款", "披露层级", "披露时点", "法定公开"].forEach((t, i) => box(s, t, 0.76 + i * 1.42, 6.1, 1.32, 0.42, { fs: 11, round: true }));
// 矩阵
const LV = { 1: ["DEEAF6", K], 2: ["BDD7EE", K], 3: ["9DC3E6", K], 4: ["2E75B6", "FFFFFF"], 5: [B2, "FFFFFF"] };
const mat = [["国家安全", 1, 4, 4, 5, 5], ["公共利益", 1, 3, 3, 4, 5], ["经济运行", 1, 2, 3, 4, 5], ["社会秩序", 1, 2, 3, 4, 5], ["其他组织", 1, 2, 2, 3, 3], ["公民个人", 1, 2, 2, 3, 3], ["单位自身", 1, 2, 2, 3, 3]];
const rows9 = [[hd("影响对象", { fontSize: 11 }), ...["无", "轻微", "一般", "严重", "特别严重"].map((h) => hd(h, { fontSize: 11 }))]];
mat.forEach((r) => rows9.push([td(r[0], { bold: true, fontSize: 11, align: "center" }), ...r.slice(1).map((v) => td(`${v} 级`, { align: "center", fontSize: 11, bold: true, fill: { color: LV[v][0] }, color: LV[v][1] }))]));
s.addTable(rows9, { x: 8.2, y: 3.3, w: 4.37, colW: [1.27, 0.5, 0.6, 0.6, 0.6, 0.8], rowH: 0.38, border: { type: "solid", pt: 0.75, color: "FFFFFF" }, objectName: nm("table") });
T(s, "南网等级判定矩阵（分级手册表 4.4）", 8.2, 6.42, 4.37, 0.3, { fontSize: 11, color: MUTE, align: "center" });
s.addNotes("阶段一完全按南网手册执行。师兄的大模型加检索增强用来解析南网规则；定级用南网的判定矩阵。多做的一步是给每个字段标注披露层级和披露时点。");

// ================= 10 阶段二① 场景 =================
s = content("阶段二 ①：确定“谁、已经知道什么”");
sub(s, "推断风险取决于接收方手里已有的数据，这一点南网文件已有规定");
// 左：嵌套层级
box(s, "", 0.76, 1.9, 5.9, 4.85, { fill: "F2F2F2", line: "A6A6A6" });
T(s, "内部数据（未披露）：调度运行、电网结构、设备运维等", 0.9, 1.95, 5.6, 0.3, { fontSize: 11.5, bold: true });
box(s, "", 1.0, 2.35, 5.42, 4.25, { fill: OR0, line: OR });
T(s, "特定信息（向特定市场成员）", 1.14, 2.4, 5.1, 0.3, { fontSize: 11.5, bold: true });
T(s, "申报量价、机组实际出力与发电量、边际能耗曲线、燃料价格与存储", 1.14, 2.72, 5.1, 0.5, { fontSize: 10.5 });
box(s, "", 1.24, 3.3, 4.94, 3.15, { fill: B0, line: B1 });
T(s, "公开信息（向有关市场成员）", 1.38, 3.35, 4.6, 0.3, { fontSize: 11.5, bold: true });
T(s, "预测类：系统负荷、发电总出力、新能源出力预测\n出清类：分时出清电价与出清量、断面阻塞\n运行类：实际负荷、机组状态、联络线潮流", 1.38, 3.68, 4.7, 0.85, { fontSize: 10.5 });
box(s, "", 1.48, 4.65, 4.46, 1.65, { fill: "FFFFFF", line: B1 });
T(s, "公众信息（向社会公众）", 1.62, 4.7, 4.2, 0.3, { fontSize: 11.5, bold: true });
T(s, "企业基本信息、装机容量、交易规则、市场运行总体情况；开放目录中无条件开放的条目；外部公开数据（气象、宏观经济等）", 1.62, 5.03, 4.2, 1.1, { fontSize: 10.5 });
T(s, "依据：《信息披露实施细则》三层级", 0.76, 6.8, 5.9, 0.28, { fontSize: 11, color: MUTE, align: "center" });
// 右上：角色表
s.addTable([
  [hd("接收方角色", { fontSize: 12 }), hd("可见层级", { fontSize: 12 }), hd("默认已知", { fontSize: 12 })],
  [td("社会公众", { bold: true }), td("1 级", { align: "center" }), td("公众信息、外部公开数据")],
  [td("协议接收方", { bold: true }), td("≤ 2 级", { align: "center" }), td("同上 + 公开信息")],
  [td("特定对象", { bold: true }), td("≤ 3 级", { align: "center" }), td("同上 + 获准的特定信息")],
], { x: 7.0, y: 1.9, w: 5.57, colW: [1.6, 1.1, 2.87], rowH: 0.46, border: BD, objectName: nm("table") });
T(s, "各角色“还可能拿到”的字段 = 等级不超过其可见层级的其他字段", 7.0, 3.85, 5.57, 0.5, { fontSize: 12, bold: true, color: RED });
// 右下：时间轴
T(s, "披露时点决定“什么时候已知”", 7.0, 4.5, 5.57, 0.3, { fontSize: 13, bold: true, color: TITLE });
[["D-1 申报前", "预测类信息"], ["D 日 交易日", "出清类信息"], ["D+1 次日", "运行类信息"]].forEach((t, i) => {
  const x = 7.0 + i * 1.93;
  sbox(s, "me", [{ text: t[0] + "\n", options: { bold: true, fontSize: 12 } }, { text: t[1] + "公开", options: { fontSize: 11 } }], x, 4.9, 1.6, 0.8, {});
  if (i < 2) arrR(s, x + 1.63, 5.17, 0.27, 0.26, GR);
});
T(s, "运行信息在 D+1 公开后，针对它的推断不再构成风险，对应分级手册附录 E 的“时效性”场景。", 7.0, 5.9, 5.57, 0.8, { fontSize: 12 });
s.addNotes("做推断风险评估先要确定场景。披露细则把市场信息分成公众、公开、特定三层，外层能看到内层的一切；分级手册规定了每一级数据可以给谁。由此得到每类接收方默认已知什么。披露时点决定什么时候已知。");

// ================= 11 阶段二② 风险量化 =================
s = content("阶段二 ②：把推断风险量化到字段");
sub(s, "从“两两关联”升级为“集合与背景条件下的推断能力”");
// 上：V(S) 流程
box(s, "字段集合 S\n（接收方可见的\n若干字段）", 0.76, 1.95, 1.9, 1.0, { fs: 11.5 });
arrR(s, 2.72, 2.3);
sbox(s, "me", "**攻击模型**\n只用 S 训练\n多种模型择优", 3.23, 1.95, 1.9, 1.0, { fs: 11.5 });
arrR(s, 5.19, 2.3);
box(s, "对敏感目标 y 的\n推断结果", 5.7, 1.95, 1.7, 1.0, { fs: 11.5, fill: GRAY, line: "A6A6A6" });
arrR(s, 7.46, 2.3);
sbox(s, "me", [{ text: "集合推断能力 V(S)\n", options: { bold: true, fontSize: 14 } }, { text: "留出数据上的推断精度 R²", options: { fontSize: 11 } }], 7.97, 1.95, 2.5, 1.0, {});
sbox(s, "sx", "师兄的 DNN / GAT 验证\n是单字段、结构目标\n时的特例", 10.67, 1.95, 1.9, 1.0, { fs: 10.5 });
// 中：三张编号卡
numCard(s, 1, "单独风险  M(0)", "这个字段自己能让目标多被推出多少。\n对应原来的单字段口径。", 0.76, 3.2, 3.75, 1.85, B1);
numCard(s, 2, "最坏背景下的风险  M(K)", "在别人已有的数据背景下，它最多能多推出多少。\n取最大值，对应“从严性”。", 4.79, 3.2, 3.75, 1.85, OR);
numCard(s, 3, "背景放大量  Γ(K)", "它的风险里有多少是被背景放大出来的。\nΓ(K) = M(K) − M(0)。", 8.82, 3.2, 3.75, 1.85, YE);
// 下：计算
T(s, "计算方式", 0.76, 5.25, 2, 0.3, { fontSize: 13, bold: true, color: TITLE });
sbox(s, "me", "**通用模型扫描**\n一个模型估计全部字段组合", 0.76, 5.62, 3.0, 0.8, { fs: 11.5 });
arrR(s, 3.83, 5.87);
sbox(s, "me", "**重训验证**\n对可见范围内的组合重训确认，\n有新发现则再次优化", 4.34, 5.62, 3.4, 0.8, { fs: 11.5 });
arrR(s, 7.81, 5.87);
box(s, "**推断证据**\n目标、见证背景（和谁一起）、\n可达精度、增量", 8.32, 5.62, 4.25, 0.8, { fs: 11.5, fill: GRAY, line: "A6A6A6" });
T(s, "已替换：师兄第四章以“字段与敏感目标的最大 MIC”作为风险输入，这里换为 M(K)。", 0.76, 6.6, 11.81, 0.35, { fontSize: 12, bold: true, color: RED });
s.addNotes("风险量化的核心是集合推断能力V：只用某个字段集合，最强的攻击模型能把目标推到多准。在它之上得到三个字段级的量。计算上先用通用模型扫描所有组合，再对当前可见范围内的组合重训验证。");

// ================= 12 例子 =================
s = content("示例：两两关联看不到的组合推断");
sub(s, "受控电力市场算例：推断某机组的私有报价加成（披露细则中的特定信息）");
s.addImage({ path: A("组合推断示例.png"), x: 0.76, y: 1.85, w: 11.81, h: 11.81 * 690 / 2252, altText: "两两关联图与组合推断对比", objectName: nm("img") });
dbox(s, 0.76, 5.65, 11.81, 1.2);
bul(s, ["每个电价单独对报价的推断能力都{{约为 0}}，机组出力单独也只有 0.065；二者合起来升到 {{0.215}}，且与调度物理一致", "电价是法定公开信息，不能收回；{{需要管住的是补上最后一步的字段}}（机组出力）", "在四个公开电力数据集上，只看单字段只能找出 {{13%~17%}} 的风险字段"], 0.95, 5.72, 11.45, 1.1, { fontSize: 13.5 });
s.addNotes("这是一个受控市场算例。左图是两两关联图，每个电价与目标之间都没有可报警的边；右图以组合为单位，机组出力和电价合起来才能反推报价。");

// ================= 13 阶段三① 一致性 =================
s = content("阶段三 ①：等级要一起定");
sub(s, "等级决定谁能拿到数据，谁能拿到又决定其他字段的风险");
// 左：回路
const L = [["字段等级", 0.9, 1.95, "nw"], ["可见范围\n（哪些角色拿得到）", 3.6, 1.95, "nw"], ["各角色的背景池", 3.6, 3.95, "me"], ["其余字段的\n推断风险 M(K)", 0.9, 3.95, "me"]];
L.forEach((b) => sbox(s, b[3], b[0], b[1], b[2], 2.1, 0.85, { fs: 12.5, bold: true }));
ln(s, 3.02, 2.37, 3.58, 2.37); ln(s, 4.65, 2.82, 4.65, 3.93); ln(s, 3.58, 4.37, 3.02, 4.37); ln(s, 1.95, 3.93, 1.95, 2.82);
T(s, "各级共享\n开放口径", 2.85, 2.45, 0.9, 0.4, { fontSize: 10, color: MUTE, align: "center" });
T(s, "上调的字段退出\n更宽角色的池", 4.75, 3.15, 1.3, 0.45, { fontSize: 10, color: MUTE });
T(s, "重算", 3.05, 4.43, 0.5, 0.25, { fontSize: 10, color: MUTE, align: "center" });
T(s, "是否仍\n需上调", 1.15, 3.15, 0.75, 0.45, { fontSize: 10, color: MUTE, align: "right" });
dbox(s, 0.76, 5.15, 5.4, 1.65);
T(s, "**一致性条件**：任何角色凭已知数据，再加上自己可见字段中的少数几个，都不能把{{高于其权限的敏感目标}}推断到泄露线以上。\n**优化目标**：满足一致性（从严性）的前提下{{上调最少}}（合理性：避免过度定级）。", 0.9, 5.2, 5.12, 1.55, { fontSize: 12.5, valign: "middle" });
// 右：示例表
T(s, "示例（示意数字）：目标 y 为 3 级，泄露线 0.7", 6.5, 1.75, 6.07, 0.3, { fontSize: 13, bold: true, color: TITLE });
const ex = [["P 出清电价", "1", "法定公开，已知", "1", false], ["A", "1", "{A,B} 0.78  {A,C} 0.74  {A,D} 0.80", "3", true], ["B", "1", "{A,B} 0.78  {B,C,D} 0.72", "1", false], ["C", "1", "{A,C} 0.74  {B,C,D} 0.72", "1", false], ["D", "2", "{A,D} 0.80  {B,C,D} 0.72", "3", true]];
const rows13 = [[hd("字段", { fontSize: 12 }), hd("规则等级", { fontSize: 12 }), hd("参与的越线组合及推断精度", { fontSize: 12 }), hd("建议等级", { fontSize: 12 })]];
ex.forEach((r) => rows13.push([td(r[0], { bold: true, align: "center" }), td(r[1], { align: "center" }), td(r[2], { fontSize: 11 }), td(r[3], { align: "center", bold: true, color: r[4] ? "FFFFFF" : K, fill: { color: r[4] ? OR : "FFFFFF" } })]));
s.addTable(rows13, { x: 6.5, y: 2.12, w: 6.07, colW: [1.25, 0.95, 2.92, 0.95], rowH: 0.42, border: BD, objectName: nm("table") });
box(s, [{ text: "全部上调\n", options: { bold: true, fontSize: 12 } }, { text: "A、B、C、D 都取目标级别\n", options: { fontSize: 11 } }, { text: "共上调 7 级次", options: { bold: true, fontSize: 15, color: MUTE } }], 6.5, 4.85, 2.9, 1.25, { fill: "F2F2F2", line: "A6A6A6" });
box(s, [{ text: "上调最少（本框架）\n", options: { bold: true, fontSize: 12 } }, { text: "只调 A、D，其余组合自然失效\n", options: { fontSize: 11 } }, { text: "共上调 3 级次", options: { bold: true, fontSize: 15, color: RED } }], 9.67, 4.85, 2.9, 1.25, { fill: OR0, line: OR });
T(s, "先调风险最大的 A → 重算 → 再调 D → 各角色可见范围内已无越线组合", 6.5, 6.3, 6.07, 0.5, { fontSize: 12, align: "center" });
s.addNotes("南网每一级都绑定了共享开放口径。字段被上调后，更宽的角色就拿不到它，其他字段的风险随之下降。所以上调谁、调几级是相互牵制的。右边的示例里，四个字段都参与了越线组合，全部上调要七级次，只调必要的字段只需三级次。表中是示意数字。");

// ================= 14 阶段三② GRPO =================
s = content("阶段三 ②：用 GRPO 求解，南网规则作硬约束");
sub(s, "沿用师兄的增强型 GRPO 算法，改造状态、动作与约束");
numCard(s, 1, "状态", "每个字段的当前等级、规则等级、M(0)、M(K)、Γ(K)、上调代价；剩余越线数", 0.76, 1.85, 2.72, 1.95, B1);
numCard(s, 2, "动作", "把某个字段上调一级，或结束", 3.79, 1.85, 2.72, 1.95, OR);
numCard(s, 3, "环境", "更新可见范围与背景池，重算受影响组合的风险", 6.82, 1.85, 2.72, 1.95, YE);
numCard(s, 4, "奖励", "越线风险的减少 − 上调代价；达到一致时给终止奖励", 9.85, 1.85, 2.72, 1.95, HEAD);
[3.5, 6.53, 9.56].forEach((x) => arrR(s, x, 2.68, 0.27, 0.26));
ln(s, 11.2, 3.82, 11.2, 4.1, { noArrow: true }); ln(s, 11.2, 4.1, 2.12, 4.1, { noArrow: true }); ln(s, 2.12, 4.1, 2.12, 3.82);
T(s, "状态随动作变化，逐步决策直到等级一致", 4.2, 4.13, 5.0, 0.28, { fontSize: 11, color: MUTE, align: "center" });
// 下：硬约束 + 变化
dbox(s, 0.76, 4.6, 5.75, 2.2, RED);
T(s, "南网硬约束（不进奖励，直接限制可选动作）", 0.9, 4.66, 5.5, 0.35, { fontSize: 13.5, bold: true, color: RED });
bul(s, ["等级不低于规则等级", "法定公开字段不可调整", "算法最高给到 3 级；涉及重要、核心数据只出申报建议", "上调不超过所涉目标的级别"], 0.95, 5.05, 5.45, 1.7, { fontSize: 12.5 });
s.addTable([
  [hd("要素", { fontSize: 12 }), hd("师兄原设定", { fontSize: 12 }), hd("本框架", { fontSize: 12 })],
  [td("风险输入", { bold: true, align: "center" }), td("字段与目标的最大 MIC"), td("M(K)、Γ(K)（集合与背景）")],
  [td("等级", { bold: true, align: "center" }), td("四级"), td("南网五级")],
  [td("合规", { bold: true, align: "center" }), td("奖励中的罚项"), td("硬约束")],
  [td("算法", { bold: true, align: "center" }), td("组内相对优势、KL 约束、熵正则"), td("不变")],
], { x: 6.82, y: 4.6, w: 5.75, colW: [1.15, 2.4, 2.2], rowH: 0.44, border: BD, objectName: nm("table") });
s.addNotes("求解沿用师兄的GRPO算法。状态里的风险量换成了本研究的M；动作是把某个字段上调一级；环境会重算风险，所以状态随动作变化。南网规则全部作为硬约束，不进奖励。右下角的表列出了相对师兄原设定的变化。");

// ================= 15 阶段四 =================
s = content("阶段四：审批固化与动态复评");
sub(s, "算法输出的是建议等级和证据，定级权在南网审批流程");
box(s, [{ text: "建议等级\n", options: { bold: true, fontSize: 13 } }, { text: "+ 推断证据", options: { fontSize: 12 } }], 0.76, 1.95, 2.2, 0.9, { fill: GR0, line: GR });
arrR(s, 3.03, 2.25);
["审批", "固化到数据目录", "技术管控", "动态复评"].forEach((t, i) => {
  s.addText(t, { x: 3.55 + i * 2.2, y: 1.95, w: 2.35, h: 0.9, shape: i === 0 ? pres.shapes.PENTAGON : pres.shapes.CHEVRON, fill: { color: i === 3 ? B2 : B1 }, line: { color: "FFFFFF", width: 1 }, fontFace: FONT, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: nm("chev") });
});
T(s, "数据目录建议新增的列", 0.76, 3.15, 5.4, 0.32, { fontSize: 14, bold: true, color: TITLE });
["规则等级", "建议等级", "上调原因", "见证背景", "可达精度", "适用角色与时点", "复评条件", "审批结论"].forEach((c, i) => box(s, c, 0.76 + (i % 2) * 2.75, 3.6 + Math.floor(i / 2) * 0.58, 2.6, 0.46, { fs: 12, round: true }));
T(s, "推断证据同时可用于风险评估手册第 382、330、351、239 项", 0.76, 6.05, 5.4, 0.6, { fontSize: 12, color: MUTE });
T(s, "复评的触发条件（事件触发，不随运行工况逐时变化）", 6.5, 3.15, 6.07, 0.32, { fontSize: 14, bold: true, color: TITLE });
s.addTable([
  [hd("触发事件", { fontSize: 12 }), hd("影响", { fontSize: 12 })],
  [td("分级手册附录 E 的场景", { bold: true }), td("体量、聚合、时效、脱敏、加工、安全事件")],
  [td("新批次开放 / 字段转为公开", { bold: true }), td("已知数据变多，其余字段的风险可能上升")],
  [td("披露时点到达", { bold: true }), td("目标本身已公开，相应的上调可以解除")],
  [td("细则修订、历史数据累积", { bold: true }), td("规则库、角色划分与推断能力需更新")],
], { x: 6.5, y: 3.6, w: 6.07, colW: [2.5, 3.57], rowH: 0.5, border: BD, objectName: nm("table") });
T(s, "复评从当前已审批的等级出发增量重算；算法不主动降级。", 6.5, 6.2, 6.07, 0.4, { fontSize: 12.5, bold: true, color: RED });
s.addNotes("算法给出的是建议等级和证据，定级仍走南网审批并固化到数据目录。动态更新是事件触发的。除了分级手册附录E的场景，新增了公开范围变化和披露时点到达两类触发。");

// ================= 16 对应表 =================
s = content("与南网文件的对应关系");
sub(s, "每个步骤都有对应条款，并处理了可能的冲突点");
s.addTable([
  [hd("体系步骤"), hd("南网条款"), hd("关系")],
  [td("分类、规则定级", { bold: true }), td("分级手册 四(一)~(三)，表 4.1~4.4，附录 A~C"), td("执行", { align: "center" })],
  [td("场景标注（角色、时点）", { bold: true }), td("披露细则 2.7、5.1~5.7；开放目录说明；注册细则"), td("引用", { align: "center" })],
  [td("推断风险识别", { bold: true }), td("分级要素“深度”；附录 D、E；风险评估第 382、330、351、239 项"), td("把原则变成可计算的方法", { align: "center" })],
  [td("取最坏背景 / 上调最少", { bold: true }), td("分级原则“从严性” / “合理性”"), td("执行", { align: "center" })],
  [td("等级与可见范围", { bold: true }), td("表 4.1~4.3 各级共享开放口径"), td("引用", { align: "center" })],
  [td("4、5 级只出申报建议", { bold: true }), td("重要数据、核心数据须按主管部门要求识别申报"), td("执行", { align: "center" })],
  [td("审批、固化、复评", { bold: true }), td("分级手册 四(四)(五)(七)，附录 E"), td("执行并具体化", { align: "center" })],
], { x: 0.76, y: 1.85, w: 11.81, colW: [3.0, 6.21, 2.6], rowH: 0.46, border: BD, objectName: nm("table") });
dbox(s, 0.76, 5.75, 11.81, 1.05);
T(s, "已处理的冲突点：{{四级改五级}}；算法{{只出建议}}，不直接定级；{{不降低规则等级}}；{{不上调法定公开字段}}；等级不随运行工况逐时变化；泄露线由目标的主管部门确定。", 0.95, 5.8, 11.45, 0.95, { fontSize: 14, valign: "middle" });
s.addNotes("这张表逐步列出了体系与南网条款的对应关系，下方是已经处理的潜在冲突点。");

// ================= 17 提纲 / 18 下一步 / 19 结束 =================
outline(3);
s = content("下一步工作");
[["1、与南网确认框架中的关键口径", "确认敏感目标清单及各自的泄露线、接收方角色与可见层级的划分、法定公开字段清单。"],
 ["2、完善等级优化与验证环节", "公开电力数据集上的初步验证表明，“上调最少”的方案比“全部上调”少调约六成；下一步完善求解方式和重训验证流程，并在试点数据上复核。"],
 ["3、开展试点", "优先选取系统运行域与电力市场披露字段，以发电企业特定信息和次日披露的运行信息为敏感目标，需要南网提供字段级历史样本。"]].forEach((it, i) => {
  const y = 1.4 + i * 1.75;
  T(s, it[0], 0.76, y, 11.8, 0.45, { fontSize: 20, bold: true, valign: "middle" });
  T(s, it[1], 1.2, y + 0.55, 11.3, 1.0, { fontSize: 17, lineSpacingMultiple: 1.25 });
});
s = pres.addSlide({ masterName: "COVER" });
T(s, "谢谢大家", 1.2, 2.0, 8, 1.6, { fontSize: 80, bold: true, color: YE, valign: "middle" });

pres.writeFile({ fileName: OUT }).then(async () => {
  // 给主题补东亚字体
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT)); const tp = "ppt/theme/theme1.xml";
  let xml = await zip.file(tp).async("string"); xml = xml.replace(/<a:ea typeface="[^"]*"\/>/g, '<a:ea typeface="微软雅黑"/>'); zip.file(tp, xml);
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("written", OUT);
});
