// V9：南方电网模板（取自 4 月进度会议汇报材料）；结构按任务书：信息归纳与结构化—一次标定—二次标定
// 运行：export PATH=/data1/duhaocun/envs/node_pptx/bin:$PATH NODE_PATH=/data1/duhaocun/envs/node_pptx_work/node_modules; node V9/scripts/build_ppt.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const V9 = path.resolve(__dirname, "..");
const A = (f) => path.join(V9, "assets", f);
const OUT = path.join(V9, "南网分类分级体系框架_V9.pptx");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const FONT = "微软雅黑";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "南网电力数据分类分级体系框架";
const TITLE = "1F4E79", LINE = "00377B", B1 = "5B9BD5", B0 = "DEEAF6", B2 = "1F4E79", NAVY = "00377B", OR = "ED7D31", OR0 = "FBE5D6",
  YE = "FFC000", GR = "70AD47", GR0 = "E2F0D9", RED = "C00000", GRAY = "E7E6E6", HEAD = "44546A", ARW = "9DC3E6", K = "000000", MUTE = "595959", CSGRED = "BA0000";
const FOOT = { text: { text: "©CSG 2020. All Rights Reserved", options: { x: 0.17, y: 7.15, w: 5, h: 0.3, fontFace: "Times New Roman", fontSize: 9, color: "A6A6A6", margin: 0 } } };
const BANNER = { image: { path: A("csg_banner.jpeg"), x: 4.62, y: 0.01, w: 8.71, h: 0.79 } };
const HLINE = { line: { x: 0, y: 0.92, w: 13.33, h: 0, line: { color: LINE, width: 1 } } };
const NUM = { x: 12.4, y: 7.1, w: 0.75, h: 0.3, fontSize: 12, color: "7F7F7F", align: "right" };
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [BANNER, HLINE,
    { placeholder: { options: { name: "title", type: "title", x: 0.95, y: 0.2, w: 8.6, h: 0.6, fontFace: FONT, fontSize: 24, bold: true, color: "2E5E9E", align: "left", valign: "middle", margin: 0 }, text: "" } }, FOOT],
  slideNumber: NUM,
});
pres.defineSlideMaster({ title: "PLAIN", background: { color: "FFFFFF" }, objects: [BANNER, HLINE, FOOT], slideNumber: NUM });
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
function content(title) { const s = pres.addSlide({ masterName: "CONTENT" }); [["B50104", 0.17], ["7F7F7F", 0.48]].forEach((c) => s.addShape(pres.shapes.CHEVRON, { x: c[1], y: 0.33, w: 0.37, h: 0.34, fill: { color: c[0] }, line: { color: c[0], width: 0 }, objectName: nm("mark") })); s.addText(title, { placeholder: "title" }); return s; }
// ---------------- V6 新增小工具 ----------------
const STAGES = ["阶段一 基础分类分级", "阶段二 推断风险识别", "阶段三 等级复核与优化", "阶段四 审批固化与复评"];
// 方法页右上的阶段导航
function nav(s, k) {
  STAGES.forEach((t, i) => box(s, t.slice(0, 3), 9.35 + i * 0.82, 1.24, 0.78, 0.3, { fill: i === k ? B2 : "F2F2F2", line: i === k ? B2 : "BFBFBF", color: i === k ? "FFFFFF" : "7F7F7F", fs: 10.5, bold: i === k }));
}
// 页脚的南网依据
function basis(s, t) {
  box(s, "南网依据", 0.76, 6.8, 0.95, 0.3, { fill: B2, line: B2, color: "FFFFFF", fs: 10.5, bold: true });
  T(s, t, 1.8, 6.8, 10.77, 0.3, { fontSize: 11, color: TITLE, valign: "middle" });
}
function method(title, k, subtitle, basisText) {
  const s = content(title);
  T(s, [{ text: "➢ ", options: { bold: true } }, ...rt(subtitle, { bold: true })], 0.76, 1.2, 8.5, 0.4, { fontSize: 18, valign: "middle" });
  nav(s, k); if (basisText) basis(s, basisText);
  return s;
}
const chev = (s, t, x, y, w, h, first, dark, fs = 12) => s.addText(t, { x, y, w, h, shape: first ? pres.shapes.PENTAGON : pres.shapes.CHEVRON, fill: { color: dark ? B2 : B1 }, line: { color: "FFFFFF", width: 1 }, fontFace: FONT, fontSize: fs, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: nm("chev") });
const head2 = (s, t, x, y, w) => T(s, t, x, y, w, 0.32, { fontSize: 14, bold: true, color: TITLE, valign: "middle" });
// 公式行：西文用 Cambria Math
const F = (s, t, x, y, w, h = 0.36, o = {}) => T(s, rt(t), x, y, w, h, { fontFace: "Cambria Math", fontSize: 15, valign: "middle", ...o });

// ---------------- V6 新增小工具 ----------------
// 带标题条的分栏框
function panel(s, title, x, y, w, h, color = B2) {
  box(s, "", x, y, w, h, { fill: "FAFCFE", line: B1, lw: 1 });
  box(s, "", x, y, w, 0.38, { fill: color, line: color });
  T(s, title, x + 0.14, y, w - 0.28, 0.38, { fontSize: 13.5, bold: true, color: "FFFFFF", valign: "middle" });
}
// 四个阶段的配色：[主色, 浅色, 主色上的文字颜色]
const SC = [[B1, B0, "FFFFFF"], [OR, OR0, "FFFFFF"], ["BF9000", "FFF2CC", "FFFFFF"], [HEAD, "D9E2F3", "FFFFFF"]];
const SNAME = ["基础分类分级", "推断风险识别", "等级复核与优化", "审批固化与复评"];
const SNUM = ["阶段一", "阶段二", "阶段三", "阶段四"];

// ---------------- V6：方法页小工具 ----------------
// 步骤标题：➢ 1.1 规则库构建：一句话说明
function stepH(s, name, desc, x, y, w, fs = 18) {
  T(s, [{ text: "➢ ", options: { bold: true } }, { text: name, options: { bold: true, color: TITLE } }, ...(desc ? [{ text: "：" + desc, options: {} }] : [])], x, y, w, 0.42, { fontSize: fs, valign: "middle" });
}
// 示例的弱化样式：灰色表头、灰色“示例”标签、灰色虚线框
const EXH = "D9D9D9", EXL = "A6A6A6", EXF = "F7F7F7";
const hdE = (t, o = {}) => ({ text: t, options: { bold: true, color: "404040", fill: { color: EXH }, fontFace: FONT, fontSize: 11, align: "center", valign: "middle", ...o } });
const tdE = (t, o = {}) => td(t, { fontSize: 11, color: "404040", ...o });
const BDE = { type: "solid", pt: 0.75, color: "D9D9D9" };
function exTag(s, x, y, t = "示例") { box(s, t, x, y, 0.62, 0.24, { fill: "7F7F7F", line: "7F7F7F", color: "FFFFFF", fs: 10, bold: true, round: true }); }
function method6(title, basisText) { const s = content(title); if (basisText) basis(s, basisText); return s; }

function outline(active) {
  const s = pres.addSlide({ masterName: "PLAIN" });
  T(s, "汇报提纲", 0.17, 0.22, 4, 0.62, { fontSize: 28, bold: true, color: "44546A", valign: "middle" });
  s.addImage({ path: A("outline_map.jpeg"), x: 2.65, y: 2.0, w: 8.24, h: 3.97, transparency: 35, objectName: nm("img") });
  ["背景与依据", "体系框架", "关键方法", "下一步工作"].forEach((t, i) => {
    const y = 2.25 + i * 0.95, on = i === active;
    s.addText(String(i + 1), { x: 4.2, y, w: 0.62, h: 0.62, shape: pres.shapes.OVAL, fill: { color: on ? "FFFFFF" : "F2F2F2" }, line: { color: on ? "7F7F7F" : "D9D9D9", width: 1.5 }, fontFace: "Arial", fontSize: 18, color: on ? K : "BFBFBF", align: "center", valign: "middle", margin: 0, objectName: nm("num") });
    ln(s, 4.9, y + 0.31, 5.5, y + 0.31, { color: on ? K : "BFBFBF", dash: "dash", width: 1.25, noArrow: true });
    T(s, t, 5.65, y, 5, 0.62, { fontSize: 24, bold: true, color: on ? K : "BFBFBF", valign: "middle" });
  });
  return s;
}
let s;
// ================= 封面 =================
function cover(main, subl, lines) {
  const c = pres.addSlide({ masterName: "PLAIN" });
  c.addShape(pres.shapes.RECTANGLE, { x: 8.7, y: 2.75, w: 4.63, h: 2.0, fill: { color: "404040" }, line: { color: "404040", width: 0 }, objectName: nm("bar") });
  c.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.9, w: 8.8, h: 2.85, fill: { color: CSGRED }, line: { color: CSGRED, width: 0 }, objectName: nm("red") });
  ln(c, 0.95, 2.2, 1.85, 2.2, { color: "FFFFFF", noArrow: true, width: 1.5 }); ln(c, 0.95, 2.2, 0.95, 4.45, { color: "FFFFFF", noArrow: true, width: 1.5 }); ln(c, 0.95, 4.45, 1.85, 4.45, { color: "FFFFFF", noArrow: true, width: 1.5 });
  T(c, main, 1.2, 2.3, 7.4, 1.6, { fontSize: 34, bold: true, color: "FFFFFF", valign: "middle" });
  if (subl) T(c, subl, 1.2, 3.95, 7.4, 0.55, { fontSize: 22, bold: true, color: "FFFFFF", valign: "middle", align: "right" });
  (lines || []).forEach((l, i) => T(c, l, 1.5, 5.15 + i * 0.6, 10.33, 0.55, { fontSize: 24, bold: true, align: "center", valign: "middle" }));
  return c;
}
cover("南网电力数据分类分级体系框架", "数据分类分级专题汇报", ["西安交通大学 网络空间学院", "2026 年 10 月"]);

// ================= 提纲 + 背景与依据 =================
outline(0);
// ================= 3 南网要求 =================
s = content("南网数据分类分级要求");
dbox(s, 0.76, 1.3, 11.81, 0.95);
T(s, "《分类分级手册》按{{影响对象}}与{{影响程度}}将数据划分为 {{1~5 级}}，多因素时{{就高不就低}}；分类细化至数据字段，分级结果经审批后固化到数据目录，并随业务与数据变化动态更新。", 0.95, 1.34, 11.45, 0.87, { fontSize: 16.5, bold: true, valign: "middle" });
head2(s, "工作流程（《分类分级手册》第四部分）", 0.76, 2.42, 6);
["数据资产梳理", "数据分类", "数据分级", "审批", "结果固化", "技术管控", "动态更新"].forEach((t, i) => chev(s, t, 0.76 + i * 1.69, 2.8, 1.78, 0.52, i === 0, false));
head2(s, "分级判定方法（《分类分级手册》四(三)、表 4.4）", 0.76, 3.55, 7);
box(s, "**分级要素（8 个）**\n领域、群体、区域、重要性\n精度、规模、覆盖度、{{深度}}", 0.76, 3.93, 2.7, 1.25, { fs: 12 });
arrR(s, 3.52, 4.4);
box(s, "**影响对象**\n群体：国家安全、公共利益、\n经济运行、社会秩序\n个体：个人、其他组织\n单位自身", 4.03, 3.93, 2.7, 1.25, { fs: 11 });
T(s, "×", 6.75, 4.35, 0.35, 0.4, { fontSize: 20, bold: true, align: "center", valign: "middle" });
box(s, "**影响程度**\n无 / 轻微 / 一般 /\n严重 / 特别严重", 7.12, 3.93, 2.0, 1.25, { fs: 12 });
arrR(s, 9.18, 4.4);
[["1 级", "DEEAF6", K], ["2 级", "BDD7EE", K], ["3 级", "9DC3E6", K], ["4 级", "2E75B6", "FFFFFF"], ["5 级", B2, "FFFFFF"]].forEach((l, i) => box(s, l[0], 9.7 + i * 0.575, 3.93, 0.575, 0.6, { fill: l[1], line: "FFFFFF", color: l[2], fs: 12, bold: true }));
box(s, "一般数据", 9.7, 4.58, 1.725, 0.6, { fill: "F2F2F2", line: "BFBFBF", fs: 12 });
box(s, "重要", 11.425, 4.58, 0.575, 0.6, { fill: "F2F2F2", line: "BFBFBF", fs: 11 });
box(s, "核心", 12.0, 4.58, 0.575, 0.6, { fill: "F2F2F2", line: "BFBFBF", fs: 11 });
head2(s, "各级数据的共享开放要求（《分类分级手册》表 4.1~4.3）", 0.76, 5.4, 7);
s.addTable([
  [hd("等级"), hd("1 级"), hd("2 级"), hd("3 级"), hd("4、5 级")],
  [td("共享开放", { bold: true, align: "center" }), td("一般对象无条件共享、开放", { align: "center" }), td("有条件开放", { align: "center" }), td("仅特定对象有条件共享、开放", { align: "center" }), td("原则上不共享、不开放；须由主管部门认定", { align: "center" })],
], { x: 0.76, y: 5.8, w: 11.81, colW: [1.4, 2.9, 1.9, 2.9, 2.71], rowH: [0.36, 0.5], border: BD, objectName: nm("table") });
s.addNotes("南网的分类分级工作手册是本体系的基线。它规定了七步工作流程、八个分级要素，以及按影响对象和影响程度判定一到五级的方法。每一级数据还对应着共享开放的要求，后面的等级优化会用到这一点。");

// ================= 4 编制依据 =================
s = content("编制依据");
T(s, [{ text: "➢ ", options: { bold: true } }, { text: "本体系依据的南网文件及其作用（后文按简称标注对应条款）", options: { bold: true } }], 0.76, 1.2, 11.8, 0.4, { fontSize: 18, valign: "middle" });
s.addTable([
  [hd("简称"), hd("文件"), hd("版本"), hd("在本体系中的作用")],
  [td("《分类分级手册》", { bold: true, align: "center", color: TITLE }), td("数据分类分级工作手册"), td("2026 年 7 月\n征求意见稿", { align: "center" }), td("分类方法、分级要素、等级判定矩阵、各级共享开放要求、审批固化与动态更新流程；{{体系基线}}")],
  [td("《风险评估手册》", { bold: true, align: "center", color: TITLE }), td("数据安全风险评估工作手册（2026 年修订版）"), td("2026 年 7 月\n征求意见稿", { align: "center" }), td("数据公开、汇聚、加工环节的风险评估项，其中第 382 项要求开展{{聚合性推断测试}}")],
  [td("《数据开放目录》", { bold: true, align: "center", color: TITLE }), td("数据开放目录（第二批）及目录说明"), td("2025 年 11 月\n征求意见稿", { align: "center" }), td("开放策略与等级的对应关系、开放对象、561 项开放数据资源")],
  [td("《披露细则》", { bold: true, align: "center", color: TITLE }), td("南方区域电力市场信息披露实施细则（2026 年 V1.0 版）"), td("2026 年 8 月\n审议稿", { align: "center" }), td("公众信息、公开信息、特定信息三类及其披露时点；用于{{确定各类接收方已掌握的信息}}")],
  [td("《注册细则》", { bold: true, align: "center", color: TITLE }), td("南方区域电力市场注册实施细则（2026 年 V1.0 版）"), td("2026 年 8 月\n审议稿", { align: "center" }), td("市场成员登记和持有的信息范围")],
], { x: 0.76, y: 1.75, w: 11.81, colW: [1.5, 3.7, 1.6, 5.01], rowH: [0.4, 0.78, 0.78, 0.7, 0.78, 0.6], border: BD, objectName: nm("table") });
dbox(s, 0.76, 6.0, 11.81, 0.75);
T(s, "上位依据：《数据安全法》《网络数据安全管理条例》《能源行业数据分类分级指南（2026 年版）》、GB/T 43697-2024《数据安全技术 数据分类分级规则》。上述南网文件为征求意见稿或审议稿，正式印发后条款编号以正式版为准。", 0.95, 6.03, 11.45, 0.69, { fontSize: 12.5, valign: "middle" });
s.addNotes("这一页列出体系依据的五份南网文件和各自的作用。后面的每一页都会用这里的简称标注对应条款。需要说明的是，这些文件目前是征求意见稿或审议稿。");

// ================= 5 问题分析 =================
s = content("问题分析：数据关联带来的推断风险");
dbox(s, 0.76, 1.3, 11.81, 0.95);
T(s, "现行分级依据的是{{数据本身泄露后的影响}}。在数据开放和共享场景下，接收方可以把新获得的数据与已掌握的数据关联起来：若干等级不高的字段组合后，{{可能推断出更高等级的数据}}。", 0.95, 1.34, 11.45, 0.87, { fontSize: 16.5, bold: true, valign: "middle" });
head2(s, "组合推断示意", 0.76, 2.45, 6.1);
[["出清电价", "1 级，法定公开"], ["机组日发电量", "1 级"], ["新能源出力预测", "1 级"]].forEach((f, i) => {
  box(s, [{ text: f[0] + "\n", options: { bold: true, fontSize: 13 } }, { text: f[1], options: { fontSize: 11, color: MUTE } }], 0.76, 2.9 + i * 0.95, 2.35, 0.75, {});
  ln(s, 3.11, 3.275 + i * 0.95, 3.75, 4.22, { color: B1, width: 1.5 });
});
box(s, "关联\n推断", 3.75, 3.82, 0.9, 0.8, { fill: OR, line: OR, color: "FFFFFF", fs: 14, bold: true, round: true });
arrR(s, 4.72, 4.07, 0.5, 0.3, OR);
box(s, [{ text: "机组报价\n", options: { bold: true, fontSize: 13, color: "FFFFFF" } }, { text: "3 级，特定信息", options: { fontSize: 11, color: "FFFFFF" } }], 5.28, 3.82, 1.6, 0.8, { fill: RED, line: RED });
T(s, "各字段单独对目标的推断能力很低；组合后推断精度可超过泄露判定线", 0.76, 5.75, 6.1, 0.4, { fontSize: 13, bold: true, color: RED, align: "center" });
head2(s, "南网文件中的相关要求", 6.95, 2.45, 5.62);
s.addTable([
  [hd("出处"), hd("要求")],
  [td("《分类分级手册》分级原则", { bold: true }), td("从严性：多因素时按可能造成的最高影响定级")],
  [td("《分类分级手册》分级要素", { bold: true }), td("深度：关联、挖掘、融合对隐含信息的刻画程度")],
  [td("《分类分级手册》附录 D", { bold: true }), td("分析挖掘出更敏感、更深度的衍生数据应提高级别")],
  [td("《分类分级手册》附录 E", { bold: true }), td("应分析数据汇聚后是否可获得更多信息")],
  [td("《风险评估手册》第 382 项", { bold: true }), td("基于已公开数据，尝试能否推断出未公开的关联信息")],
], { x: 6.95, y: 2.9, w: 5.62, colW: [2.6, 3.02], rowH: 0.52, border: BD, objectName: nm("table") });
T(s, "上述要求尚无配套的计算方法与判定标准，本体系在此处给出{{可计算、可复核}}的做法。", 6.95, 6.1, 5.62, 0.6, { fontSize: 13, bold: true });
s.addNotes("现行分级看的是数据本身泄露后的影响。但在开放共享场景下，几个等级不高的字段组合起来，可能推断出更高等级的数据。南网文件在五处提出了相关要求，但没有配套的计算方法和判定标准，这是本体系要补充的部分。");


outline(1);

// ================= 体系框架：总体结构 =================
s = content("体系框架：总体结构");
T(s, "体系组成与各步骤", 0.76, 1.2, 6, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle" });
T(s, "产出", 8.94, 1.2, 1.4, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle", align: "center" });
T(s, "南网依据", 10.46, 1.2, 2.11, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle", align: "center" });
const PARTS = [
  ["一", "信息归纳与结构化", "从业务流、数据流归纳信息类型", ["**1.1 业务流归类**\n分类到字段", "**1.2 数据流标注**\n披露层级与时点", "**1.3 异构信息结构化**\n大模型 + 检索增强"], "字段清单", "《分类分级手册》四(一)(二)\n《披露细则》2.7、5.1", 1.0],
  ["二", "一次标定", "面向本体安全威胁", ["**2.1 影响分析**\n影响对象、范围、程度", "**2.2 等级判定**\n判定矩阵，就高不就低"], "一次等级", "《分类分级手册》四(三)\n表 4.1~4.4", 1.0],
  ["三", "二次标定", "面向多源跨域融合推断威胁", ["**3.1 融合推断场景**\n接收方已掌握\n的信息", "**3.2 推断敏感度**\n字段对推断的\n最大贡献", "**3.3 计算与验证**\n关联分析、集合\n扫描、重训验证", "**3.4 二次标定**\n折算影响程度，\n查判定矩阵"], "字段推断敏感度\n二次等级", "《分类分级手册》要素“深度”、\n“从严性”、附录 D、E\n《风险评估手册》第 382 项\n《披露细则》5.2~5.7", 1.45],
  ["四", "审批固化与更新", "南网既有流程", ["**4.1 审批**", "**4.2 结果固化**\n写入数据目录", "**4.3 动态更新**\n事件触发"], "分类分级清单", "《分类分级手册》四(四)(五)(七)、\n附录 E", 0.95],
];
let py = 1.58;
PARTS.forEach((p, k) => {
  const h = p[6], c = SC[k];
  box(s, "", 0.76, py, 2.2, h, { fill: c[0], line: c[0] });
  T(s, [{ text: p[0] + "、", options: { fontSize: 13, bold: true, color: "FFFFFF" } }, { text: p[1].replace("\n", ""), options: { fontSize: p[1].length > 5 ? 13 : 15, bold: true, color: "FFFFFF" } }], 0.84, py + 0.06, 2.06, 0.5, { valign: "middle" });
  T(s, p[2], 0.84, py + 0.58, 2.06, 0.36, { fontSize: 10, color: "FFFFFF" });
  box(s, "", 2.96, py, 5.7, h, { fill: "FFFFFF", line: c[0] });
  const n = p[3].length, gap = 0.22, w = (5.5 - gap * (n - 1)) / n;
  p[3].forEach((it, i) => {
    const x = 3.06 + i * (w + gap);
    box(s, it, x, py + 0.12, w, h - 0.24, { fill: c[1], line: c[0], fs: n > 3 ? 10 : 10.5 });
    if (i < n - 1) arrR(s, x + w + 0.02, py + h / 2 - 0.1, gap - 0.04, 0.2, c[0]);
  });
  arrR(s, 8.7, py + h / 2 - 0.12, 0.2, 0.24, "A6A6A6");
  box(s, p[4], 8.94, py, 1.4, h, { fill: GRAY, line: "A6A6A6", fs: 10.5, bold: true });
  box(s, "", 10.46, py, 2.11, h, { fill: "F7F7F7", line: "A6A6A6", dash: "dash" });
  T(s, p[5], 10.52, py + 0.03, 2.0, h - 0.06, { fontSize: 8.5, valign: "middle" });
  if (k < 3) arrD(s, 1.66, py + h + 0.01, 0.4, 0.16, "A6A6A6");
  py += h + 0.19;
});
T(s, "一次标定执行南网现行规则；二次标定在一次等级的基础上，针对融合推断威胁对等级进行复核。", 0.76, 6.75, 11.81, 0.3, { fontSize: 12, bold: true, color: RED, valign: "middle" });
s.addNotes("体系按课题任务书分为三部分：信息类型归纳与结构化处理、一次标定、二次标定，最后接南网的审批固化流程。中间一列是每一部分的产出，右侧是对应的南网依据。");

// ================= 体系框架：输入、输出与约束 =================
s = content("体系框架：输入、输出与约束");
stepH(s, "体系的输入、输出与约束条件", "", 0.76, 1.25, 11.8);
box(s, "输入", 0.76, 1.95, 3.1, 0.4, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
[["南网规则文件", "五份文件的条款、表格与示例"], ["数据资产清单", "字段的业务归属、名称与描述"], ["样本数据", "字段级历史样本"], ["敏感目标", "由数据主管部门确定"]].forEach((it, i) => box(s, [{ text: it[0], options: { bold: true, fontSize: 12.5, breakLine: true } }, { text: it[1], options: { fontSize: 10.5, color: MUTE } }], 0.76, 2.45 + i * 0.82, 3.1, 0.72, { fill: "FFFFFF" }));
arrR(s, 3.95, 3.75, 0.45, 0.36);
box(s, "体系", 4.5, 1.95, 4.33, 0.4, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
["信息类型归纳与结构化处理", "一次标定", "二次标定", "审批固化与动态更新"].forEach((t, i) => { box(s, ["一", "二", "三", "四"][i], 4.5, 2.45 + i * 0.82, 0.8, 0.72, { fill: SC[i][0], line: SC[i][0], color: "FFFFFF", fs: 14, bold: true }); box(s, t, 5.3, 2.45 + i * 0.82, 3.53, 0.72, { fill: SC[i][1], line: SC[i][0], fs: 13, bold: true }); });
arrR(s, 8.92, 3.75, 0.45, 0.36);
box(s, "输出", 9.47, 1.95, 3.1, 0.4, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
[["分类分级清单", "分类路径、一次等级、二次等级"], ["字段推断敏感度表", "被推断的数据、背景、敏感度"], ["等级复核意见", "二次标定的依据说明"], ["更新记录", "触发事件、等级变动及原因"]].forEach((it, i) => box(s, [{ text: it[0], options: { bold: true, fontSize: 12.5, breakLine: true } }, { text: it[1], options: { fontSize: 10.5, color: MUTE } }], 9.47, 2.45 + i * 0.82, 3.1, 0.72, { fill: "FFFFFF" }));
dbox(s, 0.76, 5.9, 11.81, 0.8, RED);
T(s, "约束", 0.9, 5.9, 0.8, 0.8, { fontSize: 14, bold: true, color: RED, valign: "middle" });
[["二次等级不低于一次等级", "《分类分级手册》“从严性”"], ["法定公开字段等级不调整", "《披露细则》第 5 部分"], ["4、5 级只提出申报建议", "《分类分级手册》级别定义"], ["定级结论以审批为准", "《分类分级手册》四(四)"]].forEach((c, i) => T(s, [{ text: c[0], options: { bold: true, fontSize: 12.5, breakLine: true } }, { text: c[1], options: { fontSize: 10.5, color: TITLE } }], 1.7 + i * 2.72, 5.92, 2.65, 0.76, { valign: "middle", align: "center" }));
s.addNotes("这一页说明体系的输入、输出和约束。输入是南网规则文件、数据资产清单、样本数据和由主管部门确定的敏感目标。输出是分类分级清单、字段推断敏感度表、复核意见和更新记录。底部是四条约束。");

outline(2);

// ================= 一、信息类型归纳与结构化处理 =================
s = method6("一、信息类型归纳与结构化处理", "《分类分级手册》四(一)(二)、附录 A；《披露细则》2.7、5.1");
stepH(s, "1.1 业务流归类", "按企业级业务架构分类到字段", 0.76, 1.2, 11.8);
["专业域", "专业级流程组", "操作级流程", "业务对象", "数据字段"].forEach((t, i) => chev(s, t, 0.76 + i * 1.72, 1.75, 1.84, 0.5, i === 0, i === 4, 12.5));
exTag(s, 9.75, 1.62);
s.addTable([[tdE("人力资源域 → 规划与计划管理 → 人力资源综合计划业务 → 单位工作计划信息 → 生效日期、单位等", { fontSize: 10.5 })]], { x: 9.75, y: 1.9, w: 2.82, colW: [2.82], rowH: 0.62, border: BDE, fill: { color: EXF }, objectName: nm("table") });
stepH(s, "1.2 数据流标注", "标注每个字段流向谁、何时公开", 0.76, 2.75, 11.8);
s.addTable([
  [hd("标注项", { fontSize: 12 }), hd("取值", { fontSize: 12 }), hd("依据", { fontSize: 12 })],
  [td("披露层级", { bold: true, align: "center" }), td("公众信息 / 公开信息 / 特定信息 / 内部数据"), td("《披露细则》5.1", { color: TITLE })],
  [td("披露时点", { bold: true, align: "center" }), td("预测类：交易申报前；出清类：交易日；运行类：运行日次日"), td("《披露细则》2.7", { color: TITLE })],
  [td("是否法定公开", { bold: true, align: "center" }), td("法定公开字段的等级不作调整"), td("《披露细则》第 5 部分", { color: TITLE })],
], { x: 0.76, y: 3.3, w: 11.81, colW: [1.9, 6.6, 3.31], rowH: [0.38, 0.4, 0.4, 0.4], border: BD, objectName: nm("table") });
stepH(s, "1.3 异构信息结构化", "将规则文件与字段说明解析为结构化记录", 0.76, 5.05, 11.8);
box(s, [{ text: "规则文件、字段说明", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "条款、表格与文本", options: { fontSize: 10.5 } }], 0.76, 5.6, 2.5, 0.85, { fill: GRAY, line: "A6A6A6" });
arrR(s, 3.33, 5.88, 0.45, 0.3);
box(s, [{ text: "大语言模型 + 检索增强", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "按字段约束抽取并校验", options: { fontSize: 10.5 } }], 3.85, 5.6, 2.9, 0.85, {});
arrR(s, 6.82, 5.88, 0.45, 0.3);
box(s, [{ text: "结构化规则库与字段清单", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "可检索、可追溯到条款", options: { fontSize: 10.5 } }], 7.34, 5.6, 2.9, 0.85, { fill: "BDD7EE" });
s.addNotes("第一部分对应任务书里的信息类型归纳和结构化处理。从业务流角度，按企业级业务架构分类到字段；从数据流角度，标注每个字段的披露层级和披露时点；规则文件和字段说明用大语言模型加检索增强解析成结构化记录。");

// ================= 二、一次标定 =================
s = method6("二、一次标定：面向本体安全威胁", "《分类分级手册》四(三)、表 4.1~4.4、附录 C");
stepH(s, "2.1 影响分析", "数据本身泄露后的影响", 0.76, 1.25, 5.6);
[["影响对象", "群体：国家安全、公共利益、经济运行、社会秩序\n个体：个人、其他组织；单位自身"], ["影响范围", "区域、规模、覆盖度"], ["影响程度", "无 / 轻微 / 一般 / 严重 / 特别严重"]].forEach((d, i) => {
  const y = 1.95 + i * 1.05;
  box(s, d[0], 0.76, y, 1.4, 0.9, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
  box(s, d[1], 2.16, y, 4.0, 0.9, { fill: "FFFFFF", fs: 11.5 });
});
arrR(s, 6.26, 3.25, 0.4, 0.34);
T(s, "一次等级是二次标定的起点，也是等级的下限。", 0.76, 5.3, 5.4, 0.5, { fontSize: 13, valign: "middle" });
stepH(s, "2.2 等级判定", "查判定矩阵，就高不就低", 6.8, 1.25, 5.77);
const LV = { 1: ["DEEAF6", K], 2: ["BDD7EE", K], 3: ["9DC3E6", K], 4: ["2E75B6", "FFFFFF"], 5: [B2, "FFFFFF"] };
const mat = [["国家安全", 1, 4, 4, 5, 5], ["公共利益", 1, 3, 3, 4, 5], ["经济运行", 1, 2, 3, 4, 5], ["社会秩序", 1, 2, 3, 4, 5], ["其他组织", 1, 2, 2, 3, 3], ["公民个人", 1, 2, 2, 3, 3], ["单位自身", 1, 2, 2, 3, 3]];
function matrix(sl, x, y, w, rh, hi) {
  const rows = [[hd("影响对象", { fontSize: 11.5 }), ...["无", "轻微", "一般", "严重", "特别严重"].map((h) => hd(h, { fontSize: 11.5 }))]];
  mat.forEach((r, ri) => rows.push([td(r[0], { bold: true, align: "center", fontSize: 11.5 }), ...r.slice(1).map((v, ci) => { const on = hi && hi.some((q) => q[0] === ri && q[1] === ci); return td(`${v} 级`, { align: "center", bold: true, fontSize: 11.5, fill: { color: on ? OR : LV[v][0] }, color: on ? "FFFFFF" : LV[v][1] }); })]));
  const u = w / 5.5; sl.addTable(rows, { x, y, w, colW: [1.3 * u, 0.72 * u, 0.82 * u, 0.82 * u, 0.82 * u, 1.02 * u], rowH: rh, border: { type: "solid", pt: 0.75, color: "FFFFFF" }, objectName: nm("table") });
}
matrix(s, 6.8, 1.95, 5.77, 0.42);
T(s, "1~3 级为一般数据，4 级为重要数据，5 级为核心数据", 6.8, 5.4, 5.77, 0.3, { fontSize: 11.5, color: MUTE, align: "center" });
s.addNotes("一次标定面向本体安全威胁，也就是数据本身泄露的后果。按影响对象、影响范围、影响程度做分析，查南网的判定矩阵得到一次等级。这一步完全执行南网现行规则。");

// ================= 三、二次标定：总体流程 =================
s = method6("三、二次标定：面向多源跨域融合推断威胁（1/6）", "《分类分级手册》分级要素“深度”、附录 D（分析挖掘出更敏感的衍生数据应提高级别）；《风险评估手册》第 382 项");
stepH(s, "总体流程", "评估字段在融合推断中的敏感度，据此复核等级", 0.76, 1.25, 11.8);
dbox(s, 0.76, 1.85, 11.81, 0.75);
T(s, "一次等级反映数据本身泄露的后果。部分字段单独泄露影响有限，但与接收方已掌握的数据融合后，可{{推断出更高等级的数据}}，其等级需要在一次标定的基础上复核。", 0.92, 1.87, 11.5, 0.71, { fontSize: 13.5, valign: "middle" });
const FLOW = [["3.1", "融合推断场景", "各类接收方已掌握、\n可能获得的信息", "推断场景", B1], ["3.2", "字段推断敏感度", "字段在已有信息基础上\n对敏感数据推断的最大贡献", "度量定义 M", OR], ["3.3", "计算与验证", "关联分析、集合扫描、\n重训验证", "字段推断敏感度表", "BF9000"], ["3.4", "等级二次标定", "按敏感度折算影响程度，\n查判定矩阵", "二次等级", HEAD]];
FLOW.forEach((f, i) => {
  const x = 0.76 + i * 3.02;
  box(s, "", x, 2.95, 2.75, 2.3, { fill: "FFFFFF", line: f[4], lw: 1.25 });
  box(s, f[0], x, 2.95, 2.75, 0.45, { fill: f[4], line: f[4], color: "FFFFFF", fs: 15, bold: true });
  T(s, f[1], x + 0.1, 3.5, 2.55, 0.42, { fontSize: 15, bold: true, align: "center", valign: "middle" });
  T(s, f[2], x + 0.1, 3.98, 2.55, 0.8, { fontSize: 11.5, align: "center", valign: "middle" });
  if (i < 3) arrR(s, x + 2.77, 3.95, 0.23, 0.3, "A6A6A6");
  arrD(s, x + 1.17, 5.3, 0.4, 0.22, "A6A6A6");
  box(s, f[3], x, 5.57, 2.75, 0.5, { fill: GRAY, line: "A6A6A6", fs: 12, bold: true });
});
T(s, "输入：字段清单、一次等级、敏感目标及其影响对象与影响程度", 0.76, 6.25, 11.81, 0.35, { fontSize: 12, color: MUTE, valign: "middle" });
s.addNotes("二次标定面向多源跨域融合推断威胁。分四步：先确定各类接收方已经掌握什么；再定义字段推断敏感度；然后计算并验证；最后按敏感度折算影响程度，查南网判定矩阵得到二次等级。每一步下面的灰框是它的产出。");

// ================= 12 阶段二 2.1 =================
s = method6("三、二次标定（2/6）", "《披露细则》5.1~5.7、2.7；《分类分级手册》表 4.1~4.3；《数据开放目录》说明二(三)；《注册细则》5.10~5.12");
stepH(s, "3.1 融合推断场景", "确定各类接收方已掌握和可能获得的信息", 0.76, 1.25, 11.8);
const Q = 1.95;
head2(s, "① 划分信息层级", 0.76, Q, 4.1);
box(s, "", 0.76, Q + 0.42, 4.1, 3.6, { fill: "F2F2F2", line: "A6A6A6" });
T(s, "内部数据（未披露）", 0.88, Q + 0.46, 3.8, 0.28, { fontSize: 11.5, bold: true });
box(s, "", 0.96, Q + 0.8, 3.7, 3.1, { fill: OR0, line: OR });
T(s, "特定信息：申报量价、机组实际出力等", 1.08, Q + 0.84, 3.5, 0.28, { fontSize: 11.5, bold: true });
box(s, "", 1.16, Q + 1.18, 3.3, 2.6, { fill: B0, line: B1 });
T(s, "公开信息：预测类、出清类、运行类", 1.28, Q + 1.22, 3.1, 0.28, { fontSize: 11.5, bold: true });
box(s, "", 1.36, Q + 1.58, 2.9, 2.08, { fill: "FFFFFF", line: B1 });
T(s, "公众信息", 1.48, Q + 1.62, 2.7, 0.28, { fontSize: 11.5, bold: true });
T(s, "企业基本信息、交易规则、市场运行总体情况；无条件开放的数据；外部公开数据", 1.48, Q + 1.95, 2.7, 1.5, { fontSize: 11 });
T(s, "外层接收方可见内层全部信息", 0.76, Q + 4.08, 4.1, 0.28, { fontSize: 11, color: MUTE, align: "center" });
arrR(s, 4.94, Q + 2.0, 0.3, 0.32);
head2(s, "② 确定场景要素", 5.32, Q, 4.35);
s.addTable([
  [hd("接收方角色", { fontSize: 12 }), hd("可见层级", { fontSize: 12 }), hd("公开基底 B", { fontSize: 12 })],
  [td("社会公众", { bold: true }), td("1 级", { align: "center" }), td("公众信息")],
  [td("协议接收方", { bold: true }), td("≤ 2 级", { align: "center" }), td("公众 + 公开信息")],
  [td("特定对象", { bold: true }), td("≤ 3 级", { align: "center" }), td("再加获准的特定信息")],
], { x: 5.32, y: Q + 0.42, w: 4.35, colW: [1.3, 1.0, 2.05], rowH: 0.42, border: BD, objectName: nm("table") });
[["公开基底 B", "该角色已掌握的信息"], ["背景池 H", "该角色还可能获得的字段"], ["背景预算 K", "推断时在 B 之外最多使用的字段数"]].forEach((d, i) => {
  const y = Q + 2.3 + i * 0.6;
  box(s, d[0], 5.32, y, 1.3, 0.5, { fill: B2, line: B2, color: "FFFFFF", fs: 12, bold: true });
  box(s, d[1], 6.62, y, 3.05, 0.5, { fill: "FFFFFF", fs: 11.5 });
});
arrR(s, 9.75, Q + 2.0, 0.3, 0.32);
head2(s, "③ 按披露时点更新", 10.13, Q, 2.44);
[["交易申报前", "预测类信息已知"], ["交易日", "出清类信息已知"], ["运行日次日", "运行类信息已知"]].forEach((t, i) => {
  const y = Q + 0.42 + i * 1.22;
  box(s, [{ text: t[0], options: { bold: true, fontSize: 13, breakLine: true } }, { text: t[1], options: { fontSize: 11 } }], 10.13, y, 2.44, 0.92, {});
  if (i < 2) arrD(s, 11.15, y + 0.96, 0.4, 0.22);
});
s.addNotes("推断风险取决于接收方已经掌握的信息，所以先要构建推断场景。第一步按披露细则划分信息层级；第二步为每类接收方确定公开基底、背景池和背景预算；第三步按披露时点更新公开基底。");



// ================= 3.2 字段推断敏感度 =================
s = method6("三、二次标定（3/6）", "《分类分级手册》分级要素“深度”、分级原则“从严性”（取最大值）");
stepH(s, "3.2 字段推断敏感度", "字段在已有信息基础上对敏感数据推断的最大贡献", 0.76, 1.25, 11.8);
box(s, "基础量", 0.76, 1.95, 1.0, 1.0, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
box(s, "", 1.76, 1.95, 5.74, 1.0, { fill: "FFFFFF", line: B1 });
F(s, "集合推断能力  V_{y}(S)", 1.9, 1.99, 5.5, 0.4, { fontSize: 16, bold: true, color: TITLE });
T(s, "仅用字段集合 S 推断敏感数据 y 所能达到的精度（R²，0~1）", 1.9, 2.42, 5.5, 0.46, { fontSize: 11.5, valign: "middle" });
arrD(s, 3.9, 3.0, 0.44, 0.24);
box(s, "核心\n指标", 0.76, 3.3, 1.0, 1.95, { fill: OR, line: OR, color: "FFFFFF", fs: 13, bold: true });
box(s, "", 1.76, 3.3, 5.74, 1.95, { fill: OR0, line: OR, lw: 1.5 });
T(s, "字段推断敏感度", 1.9, 3.34, 5.5, 0.36, { fontSize: 15, bold: true, color: RED, valign: "middle" });
F(s, "M_{i→y}^{(K)} = max [ V_{y}(B∪T∪{i}) − V_{y}(B∪T) ]，T ⊆ H，|T| ≤ K", 1.9, 3.72, 5.5, 0.44, { fontSize: 14.5, bold: true, color: RED });
T(s, "在接收方已掌握的信息 B 和不超过 K 个其他字段 T 的基础上，再获得字段 i，对 y 的推断精度最多提高多少。取值 0~1；取到最大值的 T 称为背景。", 1.9, 4.2, 5.5, 0.95, { fontSize: 11.5, valign: "middle" });
T(s, "K 为背景字段数的上限，取 2；可逐步增大至 M 不再明显变化。", 0.76, 5.45, 6.74, 0.4, { fontSize: 12, valign: "middle" });
T(s, "与关联强度相比：M 衡量可推断的程度，并计入接收方已有的信息。", 0.76, 5.87, 6.74, 0.4, { fontSize: 12, valign: "middle" });
box(s, "", 7.85, 1.95, 4.72, 4.4, { fill: EXF, line: EXL, dash: "dash" });
exTag(s, 8.0, 2.08);
T(s, "评估字段 A 对敏感数据 y 的推断敏感度，K = 2", 8.72, 2.08, 3.75, 0.24, { fontSize: 10.5, color: "404040", valign: "middle" });
s.addTable([
  [hdE("背景"), hdE("V(背景)"), hdE("V(背景∪{A})"), hdE("提高")],
  [tdE("B", { align: "center" }), tdE("0.10", { align: "center" }), tdE("0.45", { align: "center" }), tdE("0.35", { align: "center" })],
  [tdE("B∪{C}", { align: "center" }), tdE("0.14", { align: "center" }), tdE("0.74", { align: "center" }), tdE("0.60", { align: "center" })],
  [tdE("B∪{D}", { align: "center" }), tdE("0.12", { align: "center" }), tdE("0.80", { align: "center" }), tdE("0.68", { align: "center", bold: true })],
  [tdE("B∪{C,D}", { align: "center" }), tdE("0.30", { align: "center" }), tdE("0.82", { align: "center" }), tdE("0.52", { align: "center" })],
], { x: 8.0, y: 2.45, w: 4.42, colW: [1.1, 0.95, 1.32, 1.05], rowH: [0.42, 0.38, 0.38, 0.38, 0.38], border: BDE, fill: { color: "FFFFFF" }, objectName: nm("table") });
F(s, "M_{A→y}^{(2)} = 0.68，背景为 B∪{D}", 8.1, 4.6, 4.3, 0.4, { fontSize: 14, bold: true, color: "404040" });
T(s, "字段 A 单独可将推断精度提高 0.35；接收方已掌握字段 D 时可提高 0.68。", 8.1, 5.1, 4.3, 0.9, { fontSize: 11.5, color: "404040", valign: "middle" });
s.addNotes("这一页定义字段推断敏感度。基础量是集合推断能力V：只用某个字段集合，能把敏感数据推断到多准。字段推断敏感度M是：在接收方已掌握的信息和不超过K个其他字段的基础上，再获得这个字段，推断精度最多能提高多少。右边灰色部分是示例：字段A在已掌握字段D的情况下提高最多，为0.68。");

// ================= 3.3 计算与验证 =================
s = method6("三、二次标定（4/6）", "《风险评估手册》第 382 项（聚合性推断测试）、第 330、351 项；《分类分级手册》附录 E");
stepH(s, "3.3 敏感度计算与验证", "关联分析、集合扫描、重训验证", 0.76, 1.25, 11.8);
numCard(s, 1, "关联分析", "计算字段间的线性、秩和非线性关联并融合，构建关联图，用于解释字段关系和排定计算优先级。", 0.76, 1.9, 3.75, 2.05, B1);
numCard(s, 2, "集合扫描", "用通用推断模型估计各字段组合对敏感数据的推断能力，得到每个字段的推断敏感度及其背景。", 4.79, 1.9, 3.75, 2.05, OR);
numCard(s, 3, "重训验证", "对敏感度较高的字段，按其背景组合重新训练推断模型，确认敏感度数值。", 8.82, 1.9, 3.75, 2.05, "BF9000");
arrR(s, 4.53, 2.8, 0.24, 0.26); arrR(s, 8.56, 2.8, 0.24, 0.26);
head2(s, "产出：字段推断敏感度表", 0.76, 4.25, 8);
s.addTable([
  [hd("字段", { fontSize: 11.5 }), hd("一次等级", { fontSize: 11.5 }), hd("敏感数据", { fontSize: 11.5 }), hd("敏感数据等级", { fontSize: 11.5 }), hd("接收方角色", { fontSize: 11.5 }), hd("背景", { fontSize: 11.5 }), hd("获得前精度", { fontSize: 11.5 }), hd("获得后精度", { fontSize: 11.5 }), hd("推断敏感度 M", { fontSize: 11.5 }), hd("验证方式", { fontSize: 11.5 })],
  ["A", "1 级", "y₁", "3 级", "协议接收方", "B∪{D}", "0.12", "0.80", "0.68", "重训验证"].map((v) => tdE(v, { align: "center", fill: { color: EXF } })),
], { x: 0.76, y: 4.7, w: 11.81, colW: [0.8, 1.0, 1.05, 1.3, 1.4, 1.1, 1.2, 1.2, 1.45, 1.31], rowH: [0.44, 0.46], border: BD, objectName: nm("table") });
exTag(s, 0.76, 5.72); T(s, "表中一行为示例", 1.46, 5.72, 4, 0.24, { fontSize: 10.5, color: MUTE, valign: "middle" });
s.addNotes("计算分三步。关联分析用来解释字段关系和排优先级。集合扫描用一个通用推断模型估计各字段组合的推断能力，得到每个字段的推断敏感度。对敏感度较高的字段再重训验证。产出是字段推断敏感度表。");


// ================= 3.4 等级二次标定：规则 =================
s = method6("三、二次标定（5/6）", "《分类分级手册》表 4.4、分级原则“从严性”“合理性”、附录 D；重要、核心数据须按主管部门要求识别申报");
stepH(s, "3.4 等级二次标定", "按推断敏感度折算影响程度，查判定矩阵", 0.76, 1.25, 11.8);
const RS = [["字段推断敏感度", "M（对敏感数据 y）", OR0, OR], ["影响程度折算", "y 的影响程度按 M 降档", B0, B1], ["查判定矩阵", "y 的影响对象 × 折算后的影响程度", B0, B1], ["二次等级", "与一次等级取高", "BDD7EE", B2]];
RS.forEach((r, i) => { const x = 0.76 + i * 3.02; box(s, [{ text: r[0], options: { bold: true, fontSize: 13.5, breakLine: true } }, { text: r[1], options: { fontSize: 10.5 } }], x, 1.9, 2.75, 0.85, { fill: r[2], line: r[3], lw: 1.25 }); if (i < 3) arrR(s, x + 2.77, 2.18, 0.23, 0.3); });
head2(s, "折算规则", 0.76, 3.0, 5.4);
s.addTable([
  [hd("字段推断敏感度 M", { fontSize: 12 }), hd("含义", { fontSize: 12 }), hd("影响程度", { fontSize: 12 })],
  [td("0.7 以上", { align: "center", bold: true }), td("基本决定 y 能否被推断"), td("与 y 相同", { align: "center" })],
  [td("0.4 ~ 0.7", { align: "center", bold: true }), td("贡献显著"), td("降一档", { align: "center" })],
  [td("0.1 ~ 0.4", { align: "center", bold: true }), td("有一定贡献"), td("降两档", { align: "center" })],
  [td("0.1 以下", { align: "center", bold: true }), td("可忽略"), td("不调整", { align: "center" })],
], { x: 0.76, y: 3.4, w: 5.4, colW: [1.9, 2.1, 1.4], rowH: 0.42, border: BD, objectName: nm("table") });
T(s, "分档界限需结合业务确认", 0.76, 5.55, 5.4, 0.28, { fontSize: 10.5, color: MUTE });
head2(s, "标定规则", 6.5, 3.0, 6.07);
box(s, "", 6.5, 3.4, 6.07, 0.95, { fill: OR0, line: OR, lw: 1.5 });
T(s, "二次等级 = max（一次等级，折算后查判定矩阵所得等级）", 6.5, 3.4, 6.07, 0.95, { fontSize: 15.5, bold: true, color: RED, align: "center", valign: "middle" });
bul(s, ["字段涉及多个敏感数据时，分别折算，取最高等级", "折算只降不升，二次等级不超过被推断数据的等级", "涉及重要数据、核心数据（4、5 级）时，只提出申报建议", "法定公开字段的等级不调整"], 6.65, 4.55, 5.9, 1.8, { fontSize: 12.5 });
s.addNotes("这是二次标定的规则。把被推断数据的影响程度按字段推断敏感度折算：敏感度在0.7以上，影响程度与被推断数据相同；0.4到0.7降一档；0.1到0.4降两档；0.1以下不调整。然后用被推断数据的影响对象和折算后的影响程度查判定矩阵，与一次等级取高，得到二次等级。等级仍由南网的矩阵给出。");

// ================= 3.4 等级二次标定：示例 =================
s = method6("三、二次标定（6/6）", "《分类分级手册》表 4.4、分级要素“深度”、附录 D");
stepH(s, "3.4 等级二次标定", "示例与复核意见", 0.76, 1.25, 11.8);
exTag(s, 0.76, 1.9);
T(s, "敏感数据 y₁：影响公共利益、一般影响（3 级）；y₂：影响其他组织、严重影响（3 级）", 1.46, 1.9, 11, 0.24, { fontSize: 11, color: "404040", valign: "middle" });
s.addTable([
  [hdE("字段"), hdE("一次"), hdE("M（y₁）"), hdE("M（y₂）"), hdE("折算与查表"), hdE("二次")],
  [tdE("A", { align: "center", bold: true }), tdE("1", { align: "center" }), tdE("0.68", { align: "center", bold: true }), tdE("0.05", { align: "center" }), tdE("y₁：一般 → 轻微；公共利益 × 轻微 = 3 级"), tdE("3", { align: "center", bold: true, color: RED })],
  [tdE("B", { align: "center", bold: true }), tdE("1", { align: "center" }), tdE("0.15", { align: "center" }), tdE("0.55", { align: "center", bold: true }), tdE("y₂：严重 → 一般；其他组织 × 一般 = 2 级"), tdE("2", { align: "center", bold: true, color: RED })],
  [tdE("C", { align: "center", bold: true }), tdE("1", { align: "center" }), tdE("0.06", { align: "center" }), tdE("0.08", { align: "center" }), tdE("均低于 0.1，不调整"), tdE("1", { align: "center", bold: true })],
  [tdE("D", { align: "center", bold: true }), tdE("2", { align: "center" }), tdE("0.30", { align: "center" }), tdE("0.75", { align: "center", bold: true }), tdE("y₂：严重（不降档）；其他组织 × 严重 = 3 级"), tdE("3", { align: "center", bold: true, color: RED })],
], { x: 0.76, y: 2.25, w: 7.0, colW: [0.5, 0.8, 0.95, 0.95, 3.0, 0.8], rowH: [0.42, 0.5, 0.5, 0.5, 0.5], border: BDE, fill: { color: "FFFFFF" }, objectName: nm("table") });
matrix(s, 8.0, 2.25, 4.57, 0.3, [[1, 1], [4, 2], [4, 3]]);
T(s, "判定矩阵中的对应位置（橙色）", 8.0, 4.7, 4.57, 0.26, { fontSize: 10.5, color: MUTE, align: "center" });
head2(s, "复核意见", 0.76, 5.0, 4);
box(s, "", 0.76, 5.38, 11.81, 1.28, { fill: EXF, line: EXL, dash: "dash" });
exTag(s, 0.9, 5.48);
T(s, "字段 A 一次等级为 1 级。在接收方已掌握公开信息和字段 D 的情况下，再获得字段 A 可使 3 级数据 y₁ 的推断精度提高 0.68，贡献显著。依据《分类分级手册》分级要素“深度”和附录 D，按 y₁ 的影响对象（公共利益）和折算后的影响程度（轻微）查表 4.4，建议二次等级为 3 级。", 1.65, 5.42, 10.8, 1.2, { fontSize: 11.5, color: "404040", valign: "middle" });
s.addNotes("这是一个示例。两个敏感数据都是3级。字段A对y1的推断敏感度是0.68，影响程度由一般降为轻微，公共利益乘轻微查表是3级，所以A的二次等级是3级。字段B对y2的敏感度是0.55，严重降为一般，其他组织乘一般是2级。字段C的敏感度都低于0.1，不调整。下面是据此生成的复核意见示例。");

// ================= 18 阶段四 =================
s = method6("四、审批固化与动态更新", "《分类分级手册》四(四)(五)(七)、附录 E；《风险评估手册》第 138、146、154 项");
stepH(s, "4.1 审批与 4.2 结果固化", "二次等级经审批后写入数据目录", 0.76, 1.25, 11.8);
box(s, [{ text: "二次等级", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "复核意见与推断证据", options: { fontSize: 11 } }], 0.76, 1.95, 2.3, 0.8, { fill: "FFF2CC", line: "BF9000" });
arrR(s, 3.13, 2.2);
["4.1 审批", "4.2 结果固化到数据目录", "技术管控", "4.3 动态更新"].forEach((t, i) => chev(s, t, 3.65 + i * 2.18, 1.95, 2.33, 0.8, i === 0, i === 3, 13));
T(s, "数据目录增加的列", 0.76, 3.0, 2.0, 0.42, { fontSize: 12.5, bold: true, color: TITLE, valign: "middle" });
["一次等级", "二次等级", "标定依据", "背景", "推断敏感度", "适用角色与时点", "复评条件", "审批结论"].forEach((c, i) => box(s, c, 2.75 + i * 1.235, 3.0, 1.16, 0.42, { fs: 11, round: true, fill: "FFFFFF" }));
stepH(s, "4.3 动态更新", "由事件触发，从已审批的等级出发增量重算", 0.76, 3.75, 11.8);
s.addTable([
  [hd("触发事件", { fontSize: 12 }), hd("影响", { fontSize: 12 }), hd("依据", { fontSize: 12 })],
  [td("体量变化、聚合合并、脱敏、加工、安全事件等", { bold: true }), td("一次等级与推断敏感度均可能变化"), td("《分类分级手册》附录 E", { color: TITLE })],
  [td("新批次数据开放、字段转为公开", { bold: true }), td("接收方已掌握的信息增加，其余字段的推断敏感度可能上升"), td("《数据开放目录》；《风险评估手册》第 138 项", { color: TITLE })],
  [td("披露时点到达", { bold: true }), td("敏感数据已披露，相应的上调可以解除"), td("《披露细则》2.7", { color: TITLE })],
  [td("制度修订、历史数据累积", { bold: true }), td("规则库、推断场景与推断敏感度需更新"), td("《分类分级手册》四(七)", { color: TITLE })],
], { x: 0.76, y: 4.4, w: 11.81, colW: [4.0, 4.2, 3.61], rowH: [0.4, 0.44, 0.44, 0.44, 0.44], border: BD, objectName: nm("table") });
s.addNotes("第四部分执行南网的审批和固化流程。方法给出的是二次等级，定级结论以审批为准。数据目录增加八列，记录标定依据和证据。动态更新由事件触发，表中列出了四类触发事件及其依据。");



// ================= 小结 =================
s = content("小结：与任务要求、南网文件的对应");
stepH(s, "体系各部分与课题任务、南网文件的对应关系", "", 0.76, 1.25, 11.8);
const SUM = [
  ["归纳电力信息类型，提出异构信息的结构化处理方法", "一、信息类型归纳与结构化处理", "《分类分级手册》四(一)(二)、附录 A；《披露细则》2.7、5.1"],
  ["面向本体安全威胁的安全等级一次标定方法", "二、一次标定", "《分类分级手册》四(三)、表 4.1~4.4、附录 C"],
  ["面向多源跨域融合推断安全威胁的安全等级二次标定方法", "三、二次标定：3.1 融合推断场景", "《披露细则》5.1~5.7；《分类分级手册》表 4.1~4.3；《数据开放目录》"],
  ["", "三、二次标定：3.2 字段推断敏感度　3.3 计算与验证", "《分类分级手册》要素“深度”；《风险评估手册》第 382、330、351 项"],
  ["", "三、二次标定：3.4 等级二次标定", "《分类分级手册》表 4.4、“从严性”“合理性”、附录 D"],
  ["电力信息敏感度的科学评估", "字段推断敏感度表、分类分级清单；审批固化与动态更新", "《分类分级手册》四(四)(五)(七)、附录 E"],
];
const rowsS = [[hd("课题任务要求"), hd("体系中的对应部分"), hd("南网依据")]];
SUM.forEach((r) => rowsS.push([td(r[0], { bold: true, fill: { color: r[0] ? B0 : "FFFFFF" } }), td(r[1], { bold: true }), td(r[2], { color: TITLE })]));
s.addTable(rowsS, { x: 0.76, y: 1.9, w: 11.81, colW: [3.9, 4.0, 3.91], rowH: [0.42, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56], border: BD, objectName: nm("table") });
dbox(s, 0.76, 5.95, 11.81, 0.75);
T(s, "与现行规定保持一致的处理：等级采用{{南网五级}}并由{{判定矩阵}}给出；二次等级{{不低于一次等级}}；{{法定公开字段不调整}}；重要、核心数据只提出申报建议；定级结论以审批为准。", 0.95, 5.97, 11.45, 0.71, { fontSize: 13, valign: "middle" });
s.addNotes("最后用一张表小结：左边是课题任务书的要求，中间是体系里对应的部分，右边是南网依据。下方是与现行规定保持一致的几项处理。");

outline(3);
s = content("下一步工作");
[["1、确认关键口径", "与南网确认敏感目标清单及其影响对象与影响程度、推断敏感度的分档界限、接收方角色与法定公开字段清单。"],
 ["2、完善二次标定方法", "在公开电力数据集上验证分档界限和背景字段数上限的取值；研究多个字段共同作用时的等级联合调整方法。"],
 ["3、开展试点应用", "优先选取系统运行域与电力市场披露数据开展试点，需协调提供字段级历史样本数据。"]].forEach((it, i) => {
  const y = 1.4 + i * 1.75;
  T(s, it[0], 0.76, y, 11.8, 0.45, { fontSize: 20, bold: true, valign: "middle" });
  T(s, it[1], 1.2, y + 0.55, 11.3, 1.0, { fontSize: 17, lineSpacingMultiple: 1.25 });
});
cover("谢谢大家", "", []);

pres.writeFile({ fileName: OUT }).then(async () => {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT)); const tp = "ppt/theme/theme1.xml";
  let xml = await zip.file(tp).async("string"); xml = xml.replace(/<a:ea typeface="[^"]*"\/>/g, '<a:ea typeface="微软雅黑"/>'); zip.file(tp, xml);
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("written", OUT);
});
