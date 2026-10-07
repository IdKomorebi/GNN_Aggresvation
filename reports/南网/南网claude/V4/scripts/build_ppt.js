// V4：面向南网汇报；按组内项目汇报模板的风格（蓝色标题 + 细线 + 校徽、➢ 小标题、红色关键词、虚线要点框、分层框架图、编号步骤卡片）
// 运行：export PATH=/data1/duhaocun/envs/node_pptx/bin:$PATH NODE_PATH=/data1/duhaocun/envs/node_pptx_work/node_modules; node V4/scripts/build_ppt.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const V4 = path.resolve(__dirname, "..");
const A = (f) => path.join(V4, "assets", f);
const OUT = path.join(V4, "南网分类分级体系框架_V4.pptx");

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
function content(title) { const s = pres.addSlide({ masterName: "CONTENT" }); s.addText(title, { placeholder: "title" }); return s; }
function outline(active) {
  const s = pres.addSlide({ masterName: "OUTLINE" });
  T(s, "汇报提纲", 3.0, 0.75, 7.33, 1.0, { fontSize: 40, color: "FFFFFF", align: "center", valign: "middle" });
  ln(s, 3.9, 1.25, 4.9, 1.25, { color: "FFFFFF", noArrow: true, width: 1 }); ln(s, 8.43, 1.25, 9.43, 1.25, { color: "FFFFFF", noArrow: true, width: 1 });
  ["背景与依据", "体系框架", "关键方法", "下一步工作"].forEach((t, i) => {
    const y = 2.95 + i * 1.02, on = i === active;
    s.addText(String(i + 1), { x: 4.27, y, w: 0.66, h: 0.66, shape: pres.shapes.OVAL, fill: { color: on ? NAVY : "D9D9D9" }, line: { color: on ? NAVY : "D9D9D9", width: 0 }, fontFace: "Times New Roman", fontSize: 22, bold: true, italic: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, objectName: nm("num") });
    T(s, t, 5.27, y, 6, 0.66, { fontSize: 28, bold: true, color: on ? K : "BFBFBF", valign: "middle" });
  });
  return s;
}

// ---------------- V4 新增小工具 ----------------
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

// ================= 3 南网要求 =================
s = content("南网数据分类分级要求");
dbox(s, 0.76, 1.3, 11.81, 0.95);
T(s, "《分级手册》按{{影响对象}}与{{影响程度}}将数据划分为 {{1~5 级}}，多因素时{{就高不就低}}；分类细化至数据字段，分级结果经审批后固化到数据目录，并随业务与数据变化动态更新。", 0.95, 1.34, 11.45, 0.87, { fontSize: 16.5, bold: true, valign: "middle" });
head2(s, "工作流程（《分级手册》第四部分）", 0.76, 2.42, 6);
["数据资产梳理", "数据分类", "数据分级", "审批", "结果固化", "技术管控", "动态更新"].forEach((t, i) => chev(s, t, 0.76 + i * 1.69, 2.8, 1.78, 0.52, i === 0, false));
head2(s, "分级判定方法（《分级手册》四(三)、表 4.4）", 0.76, 3.55, 7);
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
head2(s, "各级数据的共享开放要求（《分级手册》表 4.1~4.3）", 0.76, 5.4, 7);
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
  [td("《分级手册》", { bold: true, align: "center", color: TITLE }), td("数据分类分级工作手册"), td("2026 年 7 月\n征求意见稿", { align: "center" }), td("分类方法、分级要素、等级判定矩阵、各级共享开放要求、审批固化与动态更新流程；{{体系基线}}")],
  [td("《风评手册》", { bold: true, align: "center", color: TITLE }), td("数据安全风险评估工作手册（2026 年修订版）"), td("2026 年 7 月\n征求意见稿", { align: "center" }), td("数据公开、汇聚、加工环节的风险评估项，其中第 382 项要求开展{{聚合性推断测试}}")],
  [td("《开放目录》", { bold: true, align: "center", color: TITLE }), td("数据开放目录（第二批）及目录说明"), td("2025 年 11 月\n征求意见稿", { align: "center" }), td("开放策略与等级的对应关系、开放对象、561 项开放数据资源")],
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
head2(s, "南网文件中的相关要求", 7.25, 2.45, 5.32);
s.addTable([
  [hd("出处"), hd("要求")],
  [td("《分级手册》分级原则", { bold: true }), td("从严性：多因素时按可能造成的最高影响定级")],
  [td("《分级手册》分级要素", { bold: true }), td("深度：关联、挖掘、融合对隐含信息的刻画程度")],
  [td("《分级手册》附录 D", { bold: true }), td("分析挖掘出更敏感、更深度的衍生数据应提高级别")],
  [td("《分级手册》附录 E", { bold: true }), td("应分析数据汇聚后是否可获得更多信息")],
  [td("《风评手册》第 382 项", { bold: true }), td("基于已公开数据，尝试能否推断出未公开的关联信息")],
], { x: 7.25, y: 2.9, w: 5.32, colW: [2.1, 3.22], rowH: 0.52, border: BD, objectName: nm("table") });
T(s, "上述要求尚无配套的计算方法与判定标准，本体系在此处给出{{可计算、可复核}}的做法。", 7.25, 6.1, 5.32, 0.6, { fontSize: 13, bold: true });
s.addNotes("现行分级看的是数据本身泄露后的影响。但在开放共享场景下，几个等级不高的字段组合起来，可能推断出更高等级的数据。南网文件在五处提出了相关要求，但没有配套的计算方法和判定标准，这是本体系要补充的部分。");

// ================= 6 提纲 =================
outline(1);

// ================= 7 体系框架总图 =================
s = content("体系框架：总体结构");
const BX = 0.76, BWB = 9.05;
function row(yy, hh, label, items) {
  band(s, label, BX, yy, BWB, hh);
  const n = items.length, gap = 0.34, x0 = BX + 0.56, wAll = BWB - 0.42 - 0.28, w = (wAll - gap * (n - 1)) / n;
  items.forEach((it, i) => {
    const x = x0 + i * (w + gap);
    box(s, it, x, yy + 0.14, w, hh - 0.28, { fs: 11 });
    if (i < n - 1) arrR(s, x + w + 0.03, yy + hh / 2 - 0.12, gap - 0.06, 0.24);
  });
}
function rowBasis(yy, hh, t) { box(s, t, 9.95, yy, 2.62, hh, { fill: "F2F2F2", line: "A6A6A6", dash: "dash", fs: 9.5, align: "left", margin: [3, 6, 3, 8] }); }
T(s, "四个阶段及其步骤", BX, 1.2, 6, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle" });
T(s, "对应的南网依据", 9.95, 1.2, 2.62, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle", align: "center" });
row(1.55, 1.15, "阶段一", ["**1.1 规则库构建**\n解析南网规则文件", "**1.2 数据分类**\n按业务架构分类到字段", "**1.3 规则定级**\n影响对象 × 影响程度\n→ 规则等级", "**1.4 场景标注**\n披露层级、披露时点、\n是否法定公开"]);
rowBasis(1.55, 1.15, "《分级手册》四(一)~(三)、\n表 4.1~4.4、附录 A~C\n《披露细则》2.7、5.1");
arrD(s, 5.1, 2.72, 0.4, 0.2);
row(2.94, 1.15, "阶段二", ["**2.1 推断场景构建**\n接收方角色、公开基底、\n背景池", "**2.2 推断风险度量**\n集合推断能力 V\n字段风险 M、Γ", "**2.3 推断风险计算**\n关联分析\n集合扫描", "**2.4 重训验证**\n确认越线组合\n形成推断证据"]);
rowBasis(2.94, 1.15, "《分级手册》要素“深度”、\n附录 D、附录 E\n《风评手册》第 382、330 项\n《披露细则》5.2~5.7");
arrD(s, 5.1, 4.11, 0.4, 0.2);
row(4.33, 1.15, "阶段三", ["**3.1 等级一致性检查**\n各角色可见范围内\n不存在越线组合", "**3.2 等级优化**\n满足一致性的前提下\n上调幅度最小", "**3.3 复核意见生成**\n建议等级 + 推断证据"]);
rowBasis(4.33, 1.15, "《分级手册》分级原则\n“从严性”“合理性”\n表 4.1~4.3 共享开放要求\n重要、核心数据认定要求");
arrD(s, 5.1, 5.5, 0.4, 0.2);
row(5.72, 0.95, "阶段四", ["**4.1 审批**", "**4.2 结果固化**\n写入数据目录", "**4.3 动态复评**\n事件触发，增量重算"]);
rowBasis(5.72, 0.95, "《分级手册》四(四)(五)(七)、\n附录 E\n《风评手册》第 138、154 项");
T(s, "阶段一、四执行南网既有流程；阶段二、三为针对关联推断风险新增的环节，其结果以建议等级和证据的形式进入阶段四。", BX, 6.78, 11.81, 0.3, { fontSize: 11.5, bold: true, color: RED, valign: "middle" });
s.addNotes("这是体系的总体结构，共四个阶段、十四个步骤，右侧是每个阶段对应的南网依据。阶段一和阶段四执行南网已有的流程；阶段二和阶段三是针对关联推断风险新增的环节。后面的关键方法部分按这里的步骤编号逐一展开。");

// ================= 8 输入输出与约束 =================
s = content("体系框架：输入、输出与约束");
T(s, [{ text: "➢ ", options: { bold: true } }, { text: "体系的输入、输出与约束条件", options: { bold: true } }], 0.76, 1.2, 11.8, 0.4, { fontSize: 18, valign: "middle" });
// 输入
box(s, "输入", 0.76, 1.8, 2.9, 0.4, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
[["南网规则文件", "《分级手册》《风评手册》《开放目录》\n《披露细则》《注册细则》"], ["数据资产清单", "字段级元数据：所属业务对象、\n字段名称与描述"], ["样本数据", "字段级历史样本，用于推断风险计算"], ["敏感目标与泄露判定线", "由数据主管部门确定"]].forEach((it, i) => {
  box(s, [{ text: it[0] + "\n", options: { bold: true, fontSize: 12 } }, { text: it[1], options: { fontSize: 10.5 } }], 0.76, 2.28 + i * 0.86, 2.9, 0.78, {});
});
arrR(s, 3.75, 3.75, 0.5, 0.36);
// 中间
box(s, "四个阶段", 4.35, 1.8, 4.3, 0.4, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
STAGES.forEach((t, i) => { box(s, t, 4.35, 2.28 + i * 0.86, 4.3, 0.62, { fill: i === 1 || i === 2 ? "BDD7EE" : B0, fs: 13, bold: true }); if (i < 3) arrD(s, 6.32, 2.92 + i * 0.86, 0.36, 0.2); });
arrR(s, 8.75, 3.75, 0.5, 0.36);
// 输出
box(s, "输出", 9.35, 1.8, 3.22, 0.4, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
[["分类分级清单", "每个字段的分类路径、规则等级、\n建议等级及依据条款"], ["推断证据表", "被推断目标、见证背景、推断精度、\n验证方式"], ["等级复核意见", "供审批使用的文字说明"], ["复评记录", "触发事件、等级变动及原因"]].forEach((it, i) => {
  box(s, [{ text: it[0] + "\n", options: { bold: true, fontSize: 12 } }, { text: it[1], options: { fontSize: 10.5 } }], 9.35, 2.28 + i * 0.86, 3.22, 0.78, {});
});
// 约束
dbox(s, 0.76, 5.85, 11.81, 0.85, RED);
T(s, "约束", 0.9, 5.88, 0.8, 0.79, { fontSize: 14, bold: true, color: RED, valign: "middle" });
[["建议等级不低于规则等级", "《分级手册》四(三)"], ["法定公开字段等级不调整", "《披露细则》第 5 部分"], ["4、5 级只提出申报建议", "《分级手册》级别定义"], ["定级结论以审批为准", "《分级手册》四(四)"]].forEach((c, i) => {
  T(s, [{ text: c[0] + "\n", options: { bold: true, fontSize: 12.5 } }, { text: c[1], options: { fontSize: 10.5, color: TITLE } }], 1.7 + i * 2.72, 5.9, 2.65, 0.75, { valign: "middle", align: "center" });
});
basis(s, "各项约束的出处见上；输出格式与《分级手册》四(五)“结果固化到数据目录”衔接");
s.addNotes("这一页说明体系的边界。输入是南网规则文件、数据资产清单、样本数据，以及由主管部门确定的敏感目标和泄露判定线。输出是分类分级清单、推断证据表、复核意见和复评记录。底部是贯穿全程的四条约束。");

// ================= 9 提纲 =================
outline(2);

// ================= 10 阶段一 (1/2) =================
s = method("阶段一：基础分类分级（1/2）", 0, "1.1 规则库构建　　1.2 数据分类", "《分级手册》四(一)(二)、附录 A（专业域划分与分类清单示例）");
head2(s, "1.1 规则库构建：将南网规则文件解析为结构化规则", 0.76, 1.72, 8);
box(s, "南网规则文件\n（五份文件的条款、\n表格与示例）", 0.76, 2.12, 2.1, 0.95, { fill: GRAY, line: "A6A6A6", fs: 11 });
arrR(s, 2.92, 2.45);
box(s, "**大语言模型 + 检索增强**\n按字段约束抽取，输出结构化记录\n并校验完整性与一致性", 3.43, 2.12, 2.9, 0.95, { fs: 11 });
arrR(s, 6.39, 2.45);
box(s, "**结构化规则库**\n每条规则可检索、可追溯到条款", 6.9, 2.12, 2.4, 0.95, { fs: 11, fill: "BDD7EE" });
s.addTable([
  [hd("规则记录的字段", { fontSize: 11 }), hd("示例", { fontSize: 11 })],
  [td("数据类别 / 示例数据", { fontSize: 10.5 }), td("调度运行管理数据", { fontSize: 10.5 })],
  [td("影响对象 / 影响程度", { fontSize: 10.5 }), td("公共利益 / 一般影响", { fontSize: 10.5 })],
  [td("等级 / 依据条款", { fontSize: 10.5 }), td("3 级 / 《分级手册》表 4.1", { fontSize: 10.5 })],
], { x: 9.5, y: 1.72, w: 3.07, colW: [1.45, 1.62], rowH: 0.3, border: BD, objectName: nm("table") });
head2(s, "1.2 数据分类：按企业级业务架构线分类到字段", 0.76, 3.5, 8);
["专业域", "专业级流程组", "操作级流程", "业务对象", "数据字段"].forEach((t, i) => chev(s, t, 0.76 + i * 2.33, 3.9, 2.48, 0.55, i === 0, i === 4, 13));
s.addTable([
  [hd("专业域", { fontSize: 12 }), hd("专业级流程组", { fontSize: 12 }), hd("操作级流程", { fontSize: 12 }), hd("业务对象", { fontSize: 12 }), hd("数据字段", { fontSize: 12 })],
  [td("人力资源域", { align: "center" }), td("规划与计划管理", { align: "center" }), td("人力资源综合计划业务", { align: "center" }), td("单位工作计划信息", { align: "center" }), td("生效日期、单位、单位类型等", { align: "center" })],
  [td("系统运行域", { align: "center" }), td("……", { align: "center" }), td("……", { align: "center" }), td("……", { align: "center" }), td("……", { align: "center" })],
], { x: 0.76, y: 4.6, w: 11.81, colW: [1.9, 2.3, 2.7, 2.4, 2.51], rowH: [0.36, 0.42, 0.34], border: BD, objectName: nm("table") });
T(s, "分类清单示例取自《分级手册》表 A.2；共 22 个专业域（2024 年版业务架构蓝图）", 0.76, 5.76, 11.81, 0.28, { fontSize: 11, color: MUTE });
bul(s, ["内容相同或可直接换算的字段（副本字段）在分类阶段合并，按同一数据项管理", "规则库给出的是定级建议，结果需经人工确认"], 0.95, 6.08, 11.4, 0.65, { fontSize: 12.5 });
s.addNotes("阶段一的前两步。第一步用大语言模型加检索增强，把五份南网文件解析成结构化规则库，每条规则都能追溯到条款。第二步按企业级业务架构做线分类，一直分到数据字段。");

// ================= 11 阶段一 (2/2) =================
s = method("阶段一：基础分类分级（2/2）", 0, "1.3 规则定级　　1.4 场景标注", "《分级手册》四(三)、表 4.4、附录 C；《披露细则》2.7（披露时点）、5.1（信息分类）");
head2(s, "1.3 规则定级：按影响对象与影响程度判定规则等级", 0.76, 1.72, 6);
const LV = { 1: ["DEEAF6", K], 2: ["BDD7EE", K], 3: ["9DC3E6", K], 4: ["2E75B6", "FFFFFF"], 5: [B2, "FFFFFF"] };
const mat = [["国家安全", 1, 4, 4, 5, 5], ["公共利益", 1, 3, 3, 4, 5], ["经济运行", 1, 2, 3, 4, 5], ["社会秩序", 1, 2, 3, 4, 5], ["其他组织", 1, 2, 2, 3, 3], ["公民个人", 1, 2, 2, 3, 3], ["单位自身", 1, 2, 2, 3, 3]];
const rowsM = [[hd("影响对象", { fontSize: 11 }), ...["无", "轻微", "一般", "严重", "特别严重"].map((h) => hd(h, { fontSize: 11 }))]];
mat.forEach((r) => rowsM.push([td(r[0], { bold: true, fontSize: 11, align: "center" }), ...r.slice(1).map((v) => td(`${v} 级`, { align: "center", fontSize: 11, bold: true, fill: { color: LV[v][0] }, color: LV[v][1] }))]));
s.addTable(rowsM, { x: 0.76, y: 2.12, w: 5.4, colW: [1.3, 0.72, 0.8, 0.8, 0.8, 0.98], rowH: 0.38, border: { type: "solid", pt: 0.75, color: "FFFFFF" }, objectName: nm("table") });
T(s, "等级判定矩阵（《分级手册》表 4.4）", 0.76, 5.2, 5.4, 0.28, { fontSize: 11, color: MUTE, align: "center" });
bul(s, ["群体、个体、单位自身三个维度分别判定，取最高者作为{{规则等级}}", "1~3 级为一般数据，4 级为重要数据，5 级为核心数据"], 0.95, 5.55, 5.3, 1.1, { fontSize: 12.5 });
head2(s, "1.4 场景标注：为推断风险识别准备的字段属性", 6.6, 1.72, 6);
s.addTable([
  [hd("标注项", { fontSize: 12 }), hd("取值", { fontSize: 12 }), hd("依据", { fontSize: 12 })],
  [td("披露层级", { bold: true, align: "center" }), td("公众信息 / 公开信息 / 特定信息 / 内部数据"), td("《披露细则》5.1", { color: TITLE })],
  [td("披露时点", { bold: true, align: "center" }), td("预测类：交易申报前\n出清类：交易日\n运行类：运行日次日"), td("《披露细则》2.7", { color: TITLE })],
  [td("是否法定公开", { bold: true, align: "center" }), td("是 / 否；法定公开字段的等级不作调整"), td("《披露细则》第 5 部分", { color: TITLE })],
  [td("是否敏感目标", { bold: true, align: "center" }), td("特定信息、规则等级 3 级及以上的数据"), td("《分级手册》表 4.1~4.3", { color: TITLE })],
], { x: 6.6, y: 2.12, w: 5.97, colW: [1.35, 2.82, 1.8], rowH: [0.36, 0.5, 0.88, 0.6, 0.6], border: BD, objectName: nm("table") });
dbox(s, 6.6, 5.3, 5.97, 1.3);
T(s, "阶段一的产出：字段清单，每个字段带有{{分类路径、规则等级、依据条款}}，以及披露层级、披露时点、法定公开标记。规则等级是后续所有调整的下限。", 6.75, 5.34, 5.67, 1.22, { fontSize: 13, valign: "middle" });
s.addNotes("第三步用南网的判定矩阵得到规则等级。第四步是为下一阶段准备的：给每个字段标注披露层级、披露时点、是否法定公开，以及是否属于敏感目标。");

// ================= 12 阶段二 2.1 =================
s = method("阶段二：推断风险识别（1/3）", 1, "2.1 推断场景构建", "《披露细则》5.1~5.7（信息分类）、2.7（披露时点）；《分级手册》表 4.1~4.3；《开放目录》说明二(三)；《注册细则》5.10~5.12");
head2(s, "信息的披露层级（外层接收方可见内层全部信息）", 0.76, 1.72, 6);
box(s, "", 0.76, 2.1, 5.9, 4.5, { fill: "F2F2F2", line: "A6A6A6" });
T(s, "内部数据（未披露）：调度运行、电网结构、设备运维等", 0.9, 2.14, 5.6, 0.28, { fontSize: 11.5, bold: true });
box(s, "", 1.0, 2.5, 5.42, 3.95, { fill: OR0, line: OR });
T(s, "特定信息（向特定市场成员披露）", 1.14, 2.54, 5.1, 0.28, { fontSize: 11.5, bold: true });
T(s, "申报量价、机组实际出力与发电量、边际能耗曲线、燃料价格与存储", 1.14, 2.84, 5.1, 0.3, { fontSize: 10.5 });
box(s, "", 1.24, 3.25, 4.94, 3.05, { fill: B0, line: B1 });
T(s, "公开信息（向有关市场成员披露）", 1.38, 3.29, 4.6, 0.28, { fontSize: 11.5, bold: true });
T(s, "预测类：系统负荷、发电总出力、新能源出力预测\n出清类：分时出清电价与出清量、断面阻塞\n运行类：实际负荷、机组状态、联络线潮流", 1.38, 3.6, 4.7, 0.8, { fontSize: 10.5 });
box(s, "", 1.48, 4.52, 4.46, 1.63, { fill: "FFFFFF", line: B1 });
T(s, "公众信息（向社会公众披露）", 1.62, 4.56, 4.2, 0.28, { fontSize: 11.5, bold: true });
T(s, "企业基本信息、装机容量、交易规则、市场运行总体情况；《开放目录》中无条件开放的数据；外部公开数据（气象、宏观经济等）", 1.62, 4.88, 4.2, 1.1, { fontSize: 10.5 });
head2(s, "推断场景的三个要素", 7.0, 1.72, 5.57);
s.addTable([
  [hd("接收方角色", { fontSize: 12 }), hd("可见层级", { fontSize: 12 }), hd("公开基底 B（已掌握）", { fontSize: 12 })],
  [td("社会公众", { bold: true }), td("1 级", { align: "center" }), td("公众信息、外部公开数据")],
  [td("协议接收方", { bold: true }), td("≤ 2 级", { align: "center" }), td("同上 + 公开信息")],
  [td("特定对象", { bold: true }), td("≤ 3 级", { align: "center" }), td("同上 + 获准的特定信息")],
], { x: 7.0, y: 2.1, w: 5.57, colW: [1.45, 1.05, 3.07], rowH: 0.42, border: BD, objectName: nm("table") });
bul(s, ["**背景池 H**：该角色还可能获得的字段，即等级不超过其可见层级的其他字段", "**背景预算 K**：推断时在公开基底之外最多使用的字段数，取 K = 2"], 7.15, 3.9, 5.4, 1.0, { fontSize: 12 });
head2(s, "公开基底随披露时点变化", 7.0, 4.95, 5.57);
[["申报前", "预测类信息"], ["交易日", "出清类信息"], ["运行日次日", "运行类信息"]].forEach((t, i) => {
  const x = 7.0 + i * 1.93;
  box(s, [{ text: t[0] + "\n", options: { bold: true, fontSize: 12 } }, { text: t[1] + "进入 B", options: { fontSize: 11 } }], x, 5.33, 1.6, 0.7, {});
  if (i < 2) arrR(s, x + 1.63, 5.56, 0.27, 0.24);
});
T(s, "运行类信息披露后，以其为目标的推断不再构成风险（《分级手册》附录 E“时效性”）。", 7.0, 6.12, 5.57, 0.5, { fontSize: 11.5 });
s.addNotes("推断风险取决于接收方已经掌握的信息，所以先要构建推断场景。披露细则把市场信息分为公众、公开、特定三类；分级手册规定了各级数据可以提供给谁。据此定义三个要素：接收方角色及其公开基底、背景池、背景预算。公开基底还会随披露时点变化。");

// ================= 13 阶段二 2.2 =================
s = method("阶段二：推断风险识别（2/3）", 1, "2.2 推断风险度量", "《分级手册》分级要素“深度”、分级原则“从严性”（取最大值）；泄露判定线由数据主管部门确定");
head2(s, "度量定义（y 为敏感目标，i 为待评估字段）", 0.76, 1.72, 7);
const defs = [
  ["集合推断能力", "V_{y}(S)", "仅用字段集合 S 训练推断模型，在留出数据上对 y 的推断精度（决定系数 R²，取值 0~1）"],
  ["边际推断增益", "Δ_{i}(T) = V_{y}(B∪T∪{i}) − V_{y}(B∪T)", "在公开基底 B 和背景字段 T 之上，再获得字段 i 带来的推断精度提升"],
  ["单字段风险", "M_{i}^{(0)} = Δ_{i}(∅)", "不使用背景字段时字段 i 的边际推断增益"],
  ["背景条件下的最大风险", "M_{i}^{(K)} = max Δ_{i}(T)，T ⊆ H，|T| ≤ K", "在所有不超过 K 个背景字段的组合中取最大值；取到最大值的 T 称为见证背景"],
  ["背景放大量", "Γ_{i}^{(K)} = M_{i}^{(K)} − M_{i}^{(0)}", "字段风险中由背景关联放大的部分"],
  ["泄露判定线", "τ_{y}", "目标 y 的推断精度超过 τ_{y} 即视为泄露；超过判定线的字段组合称为越线组合"],
];
defs.forEach((d, i) => {
  const y = 2.1 + i * 0.75;
  box(s, d[0], 0.76, y, 1.75, 0.65, { fill: B2, line: B2, color: "FFFFFF", fs: 11.5, bold: true });
  box(s, "", 2.51, y, 5.39, 0.65, { fill: "FFFFFF", line: B1 });
  F(s, d[1], 2.62, y + 0.02, 5.2, 0.3, { fontSize: 14, bold: true, color: TITLE });
  T(s, rt(d[2]), 2.62, y + 0.33, 5.2, 0.3, { fontSize: 10, valign: "middle" });
});
head2(s, "计算示例（示意数值，K = 2，τ = 0.7）", 8.2, 1.72, 4.37);
s.addTable([
  [hd("已掌握的字段", { fontSize: 11 }), hd("未获得 A", { fontSize: 11 }), hd("获得 A 后", { fontSize: 11 }), hd("增益", { fontSize: 11 })],
  [td("仅公开基底 B", { fontSize: 11 }), td("0.10", { align: "center" }), td("0.45", { align: "center" }), td("0.35", { align: "center", bold: true })],
  [td("B + 字段 C", { fontSize: 11 }), td("0.14", { align: "center" }), td("0.74", { align: "center" }), td("0.60", { align: "center" })],
  [td("B + 字段 D", { fontSize: 11 }), td("0.12", { align: "center" }), td("0.80", { align: "center", color: RED, bold: true }), td("0.68", { align: "center", bold: true, color: RED })],
], { x: 8.2, y: 2.1, w: 4.37, colW: [1.42, 0.95, 1.0, 1.0], rowH: 0.4, border: BD, objectName: nm("table") });
T(s, "表中为对目标 y 的推断精度 V", 8.2, 3.73, 4.37, 0.25, { fontSize: 10.5, color: MUTE, align: "center" });
box(s, "", 8.2, 4.1, 4.37, 1.45, {});
T(s, [{ text: "字段 A 的度量结果", options: { bold: true, fontSize: 12.5, breakLine: true } }, { text: "单字段风险 M(0) = 0.35", options: { fontSize: 12, breakLine: true } }, { text: "背景条件下的最大风险 M(2) = 0.68，见证背景为字段 D", options: { fontSize: 12, breakLine: true } }, { text: "背景放大量 Γ(2) = 0.33", options: { fontSize: 12 } }], 8.38, 4.16, 4.05, 1.33, { valign: "middle" });
dbox(s, 8.2, 5.7, 4.37, 0.92, RED);
T(s, "判定：已掌握字段 D 的接收方再获得字段 A 后，推断精度由 0.12 升至 0.80，{{超过判定线 0.7}}，{A, D} 为越线组合。", 8.32, 5.73, 4.13, 0.86, { fontSize: 11.5, valign: "middle" });
s.addNotes("这一页给出度量的定义。核心量是集合推断能力V：只用某个字段集合，能把敏感目标推断到多准。在它之上定义边际推断增益，再得到单字段风险M零、背景条件下的最大风险MK和背景放大量。右边是一个计算示例：字段A单独的增益是0.35，在已掌握字段D时增益达到0.68，并且使推断精度超过了判定线。");

// ================= 14 阶段二 2.3 2.4 =================
s = method("阶段二：推断风险识别（3/3）", 1, "2.3 推断风险计算　　2.4 重训验证", "《风评手册》第 382 项（聚合性推断测试）、第 330 项（汇聚）、第 351 项（加工产生新数据）；《分级手册》附录 E");
numCard(s, 1, "关联分析", "计算字段间的线性相关、秩相关和非线性关联，融合为关联强度，构建有向加权关联图。\n用途：解释字段关系、识别副本字段、排定计算优先级。", 0.76, 1.75, 3.75, 2.25, B1);
numCard(s, 2, "集合扫描", "训练一个通用推断模型，对公开基底之上所有不超过 K+1 个字段的组合估计推断能力 V，得到各字段的 M(0)、M(K)、Γ(K) 和候选见证背景。", 4.79, 1.75, 3.75, 2.25, OR);
numCard(s, 3, "重训验证", "对接收方可见范围内的字段组合，用多种推断模型、全量样本逐一重训，确认是否越线；发现新的越线组合后重新优化，直至无新增。", 8.82, 1.75, 3.75, 2.25, YE);
arrR(s, 4.53, 2.75, 0.24, 0.26); arrR(s, 8.56, 2.75, 0.24, 0.26);
T(s, "说明：关联分析只用于辅助，不作为筛除依据（单字段关联弱的字段仍可能是越线组合的成员）；集合扫描给出估计值，定级依据以重训验证的结果为准。", 0.76, 4.1, 11.81, 0.5, { fontSize: 12, bold: true, color: RED, valign: "middle" });
head2(s, "产出：推断证据表（每条越线组合一行）", 0.76, 4.72, 8);
s.addTable([
  [hd("敏感目标", { fontSize: 11 }), hd("目标等级", { fontSize: 11 }), hd("字段", { fontSize: 11 }), hd("接收方角色", { fontSize: 11 }), hd("见证背景", { fontSize: 11 }), hd("获得前精度", { fontSize: 11 }), hd("获得后精度", { fontSize: 11 }), hd("M(0)", { fontSize: 11 }), hd("M(K)", { fontSize: 11 }), hd("判定线", { fontSize: 11 }), hd("验证方式", { fontSize: 11 })],
  [td("y", { align: "center" }), td("3 级", { align: "center" }), td("A", { align: "center" }), td("协议接收方", { align: "center" }), td("D", { align: "center" }), td("0.12", { align: "center" }), td("0.80", { align: "center" }), td("0.35", { align: "center" }), td("0.68", { align: "center" }), td("0.7", { align: "center" }), td("重训验证", { align: "center" })],
], { x: 0.76, y: 5.1, w: 11.81, colW: [1.05, 0.95, 0.75, 1.3, 1.1, 1.15, 1.15, 0.85, 0.85, 0.9, 1.8], rowH: [0.4, 0.42], border: BD, objectName: nm("table") });
T(s, "示例行为示意数值。推断证据表同时可作为《风评手册》第 382 项等评估项的评估材料。", 0.76, 6.05, 11.81, 0.3, { fontSize: 11.5, color: MUTE });
s.addNotes("计算分三步。关联分析用来解释字段关系和排优先级，不作为筛除依据。集合扫描用一个通用推断模型估计所有字段组合的推断能力。重训验证对可见范围内的组合逐一重训确认，定级依据以验证结果为准。产出是推断证据表，每条越线组合一行。");

// ================= 16 阶段三 3.1 =================
s = method("阶段三：等级复核与优化（1/3）", 2, "3.1 等级一致性检查", "《分级手册》表 4.1~4.3（各级数据的共享开放要求）、分级原则“从严性”");
head2(s, "等级、可见范围与推断风险的关系", 0.76, 1.72, 5.6);
const L = [["字段等级", 0.95, 2.2], ["可见范围\n（可获得该字段的角色）", 3.75, 2.2], ["各角色的背景池", 3.75, 4.1], ["其余字段的推断风险", 0.95, 4.1]];
L.forEach((b, i) => box(s, b[0], b[1], b[2], 2.25, 0.85, { fs: 12.5, bold: true, fill: i < 2 ? B0 : "BDD7EE" }));
ln(s, 3.22, 2.62, 3.73, 2.62); ln(s, 4.87, 3.07, 4.87, 4.08); ln(s, 3.73, 4.52, 3.22, 4.52); ln(s, 2.07, 4.08, 2.07, 3.07);
T(s, "共享开放要求", 2.9, 3.1, 1.2, 0.25, { fontSize: 10, color: MUTE, align: "center" });
T(s, "等级上调的字段\n退出较宽角色的背景池", 4.97, 3.33, 1.6, 0.45, { fontSize: 10, color: MUTE });
T(s, "重新计算", 3.0, 4.98, 0.95, 0.25, { fontSize: 10, color: MUTE, align: "center" });
T(s, "是否仍需\n上调", 1.2, 3.33, 0.8, 0.45, { fontSize: 10, color: MUTE, align: "right" });
T(s, "每一级数据对应确定的共享开放范围。一个字段的等级上调后，可获得它的角色减少，其他字段在这些角色手中的推断风险随之下降，因此各字段的等级需要统一考虑。", 0.76, 5.2, 5.6, 1.3, { fontSize: 12.5 });
head2(s, "等级一致性条件", 6.75, 1.72, 5.82);
dbox(s, 6.75, 2.1, 5.82, 2.05);
T(s, "对任一接收方角色 r 和任一等级高于其可见层级的敏感目标 y，该角色凭公开基底，加上其背景池中任意不超过 K+1 个字段，对 y 的推断精度都不超过泄露判定线：", 6.9, 2.14, 5.52, 0.95, { fontSize: 12.5, valign: "middle" });
F(s, "V_{y}(B_{r} ∪ S) ≤ τ_{y}，  ∀ S ⊆ H_{r}(ℓ)，|S| ≤ K+1", 6.9, 3.1, 5.52, 0.42, { fontSize: 16, bold: true, color: TITLE, align: "center" });
F(s, "H_{r}(ℓ) = { i : ℓ_{i} ≤ c_{r} }  为角色 r 在当前等级 ℓ 下的背景池，c_{r} 为其可见层级", 6.9, 3.58, 5.52, 0.42, { fontSize: 11.5, align: "center" });
s.addTable([
  [hd("检查结果", { fontSize: 12 }), hd("处理", { fontSize: 12 })],
  [td("各角色背景池内均无越线组合", { bold: true }), td("等级一致，规则等级即为建议等级")],
  [td("存在越线组合", { bold: true }), td("进入 3.2 等级优化，确定需要上调的字段")],
  [td("仅凭公开基底即可越线", { bold: true }), td("问题出在已公开的数据，单独提交主管部门研判")],
], { x: 6.75, y: 4.35, w: 5.82, colW: [2.55, 3.27], rowH: [0.36, 0.48, 0.48, 0.55], border: BD, objectName: nm("table") });
s.addNotes("南网每一级数据都对应确定的共享开放范围。一个字段的等级上调后，能拿到它的角色就少了，其他字段的推断风险随之下降。所以各字段的等级需要统一考虑。右边是等级一致性条件的定义，以及检查后的三种处理方式。");

// ================= 17 阶段三 3.2 模型 =================
s = method("阶段三：等级复核与优化（2/3）", 2, "3.2 等级优化：模型与示例", "《分级手册》分级原则“从严性”（消除全部越线组合）、“合理性”（避免过度定级）；级别定义（重要、核心数据须认定）");
head2(s, "优化模型", 0.76, 1.72, 5.4);
dbox(s, 0.76, 2.1, 5.4, 3.0);
T(s, "目标：上调幅度最小", 0.9, 2.15, 5.1, 0.3, { fontSize: 12.5, bold: true });
F(s, "min  Σ_{i} w_{i} · ( ℓ_{i} − ℓ_{i}^{规则} )", 0.9, 2.45, 5.1, 0.42, { fontSize: 16, bold: true, color: TITLE, align: "center" });
T(s, "约束：", 0.9, 2.92, 5.1, 0.28, { fontSize: 12.5, bold: true });
[["ℓ_{i} ≥ ℓ_{i}^{规则}", "建议等级不低于规则等级"], ["ℓ_{i} = ℓ_{i}^{规则}（i 为法定公开字段）", "法定公开字段不调整"], ["ℓ_{i} ≤ 3", "涉及 4、5 级时只提出申报建议"], ["等级一致性条件成立", "不存在越线组合"]].forEach((c, i) => {
  F(s, c[0], 1.05, 3.22 + i * 0.44, 2.75, 0.4, { fontSize: 12.5 });
  T(s, c[1], 3.85, 3.22 + i * 0.44, 2.25, 0.4, { fontSize: 11, color: MUTE, valign: "middle" });
});
T(s, "w_{i} 为字段 i 每上调一级的代价（对数据使用的影响），可由业务部门给出，缺省取 1。", 0.76, 5.2, 5.4, 0.55, { fontSize: 11.5 });
T(s, "上调任一字段只会减少越线组合，不会产生新的越线组合；将所有涉及字段都上调到目标等级是一个可行解，但上调幅度最大。", 0.76, 5.8, 5.4, 0.8, { fontSize: 11.5 });
head2(s, "示例（示意数值）：目标 y 为 3 级，判定线 0.7", 6.5, 1.72, 6.07);
const ex = [["P（出清电价）", "1", "法定公开，属公开基底", "1", false], ["A", "1", "{A,B} 0.78  {A,C} 0.74  {A,D} 0.80", "3", true], ["B", "1", "{A,B} 0.78  {B,C,D} 0.72", "1", false], ["C", "1", "{A,C} 0.74  {B,C,D} 0.72", "1", false], ["D", "2", "{A,D} 0.80  {B,C,D} 0.72", "3", true]];
const rowsE = [[hd("字段", { fontSize: 12 }), hd("规则等级", { fontSize: 12 }), hd("所在越线组合及推断精度", { fontSize: 12 }), hd("建议等级", { fontSize: 12 })]];
ex.forEach((r) => rowsE.push([td(r[0], { bold: true, align: "center" }), td(r[1], { align: "center" }), td(r[2], { fontSize: 11 }), td(r[3], { align: "center", bold: true, color: r[4] ? "FFFFFF" : K, fill: { color: r[4] ? OR : "FFFFFF" } })]));
s.addTable(rowsE, { x: 6.5, y: 2.1, w: 6.07, colW: [1.45, 0.95, 2.72, 0.95], rowH: 0.42, border: BD, objectName: nm("table") });
box(s, [{ text: "全部上调\n", options: { bold: true, fontSize: 12 } }, { text: "A、B、C、D 均上调至 3 级\n", options: { fontSize: 11 } }, { text: "合计上调 7 级", options: { bold: true, fontSize: 15, color: MUTE } }], 6.5, 4.8, 2.9, 1.2, { fill: "F2F2F2", line: "A6A6A6" });
box(s, [{ text: "优化结果\n", options: { bold: true, fontSize: 12 } }, { text: "仅上调 A、D，其余组合随之消除\n", options: { fontSize: 11 } }, { text: "合计上调 3 级", options: { bold: true, fontSize: 15, color: RED } }], 9.67, 4.8, 2.9, 1.2, { fill: OR0, line: OR });
T(s, "上调 A 后重新计算，剩余越线组合 {B,C,D}；再上调 D 一级后各角色背景池内无越线组合。", 6.5, 6.1, 6.07, 0.5, { fontSize: 11.5 });
s.addNotes("等级优化的模型：目标是上调幅度最小，约束包括不低于规则等级、法定公开字段不调整、涉及四五级时只提申报建议，以及等级一致性条件成立。右边的示例里，四个字段都处在越线组合中，全部上调合计七级，优化后只需上调A和D，合计三级。表中是示意数值。");

// ================= 18 阶段三 3.2 求解 + 3.3 =================
s = method("阶段三：等级复核与优化（3/3）", 2, "3.2 等级优化：求解　　3.3 复核意见生成", "《分级手册》四(三)(四)、分级要素“深度”、附录 D（衍生数据分级参考）、表 4.4");
head2(s, "基于强化学习（GRPO）的序贯求解", 0.76, 1.72, 8);
numCard(s, 1, "状态", "各字段的当前等级、规则等级、M(0)、M(K)、Γ(K)、上调代价；剩余越线组合数", 0.76, 2.1, 2.72, 1.75, B1);
numCard(s, 2, "动作", "选择一个字段上调一级，或结束", 3.79, 2.1, 2.72, 1.75, OR);
numCard(s, 3, "环境", "更新各角色的背景池，重新计算受影响组合的推断风险", 6.82, 2.1, 2.72, 1.75, YE);
numCard(s, 4, "奖励", "越线风险的减少量减去上调代价；达到等级一致时给予终止奖励", 9.85, 2.1, 2.72, 1.75, HEAD);
[3.5, 6.53, 9.56].forEach((x) => arrR(s, x, 2.85, 0.27, 0.26));
ln(s, 11.2, 3.87, 11.2, 4.08, { noArrow: true }); ln(s, 11.2, 4.08, 2.12, 4.08, { noArrow: true }); ln(s, 2.12, 4.08, 2.12, 3.87);
T(s, "逐步决策，直至等级一致", 4.2, 4.1, 5.0, 0.26, { fontSize: 10.5, color: MUTE, align: "center" });
bul(s, ["策略更新采用组内相对优势估计，并加入策略偏移约束与熵正则，保证更新稳定", "3.2 中的各项约束以限制可选动作的方式实现，不作为奖励中的惩罚项", "规模较小时可用整数规划直接求解，作为对照"], 0.95, 4.48, 6.0, 1.9, { fontSize: 12 });
head2(s, "3.3 复核意见示例（示意数值）", 7.2, 4.42, 5.37);
dbox(s, 7.2, 4.8, 5.37, 1.85);
T(s, "字段 A 规则等级为 1 级。经推断风险识别，协议接收方在已掌握出清电价和字段 D 的情况下再获得字段 A，可将 3 级数据 y 的推断精度由 0.12 提高到 0.80，超过泄露判定线 0.7（经重训验证）。依据《分级手册》分级要素“深度”和附录 D，字段 A 泄露后的影响程度参照 y 评价，按表 4.4 {{建议复核为 3 级}}。", 7.32, 4.83, 5.13, 1.79, { fontSize: 11, valign: "middle" });
s.addNotes("求解采用强化学习的GRPO算法，逐步决定上调哪个字段。状态是各字段的等级和风险度量，动作是选一个字段上调一级，环境重新计算风险，奖励是越线风险的减少量减去上调代价。各项约束通过限制可选动作实现。规模较小时也可以用整数规划直接求解作为对照。右下是生成的复核意见示例。");

// ================= 19 阶段四 =================
s = method("阶段四：审批固化与动态复评", 3, "4.1 审批　　4.2 结果固化　　4.3 动态复评", "《分级手册》四(四)(五)(七)、附录 E；《风评手册》第 138、146、154 项");
box(s, [{ text: "建议等级\n", options: { bold: true, fontSize: 13 } }, { text: "复核意见与推断证据", options: { fontSize: 11.5 } }], 0.76, 1.8, 2.3, 0.85, { fill: "BDD7EE" });
arrR(s, 3.13, 2.07);
["4.1 审批", "4.2 结果固化到数据目录", "技术管控", "4.3 动态复评"].forEach((t, i) => chev(s, t, 3.65 + i * 2.18, 1.8, 2.33, 0.85, i === 0, i === 3, 13));
head2(s, "数据目录中建议增加的列", 0.76, 2.95, 5.4);
["规则等级", "建议等级", "上调原因", "见证背景", "推断精度", "适用角色与时点", "复评条件", "审批结论"].forEach((c, i) => box(s, c, 0.76 + (i % 2) * 2.75, 3.35 + Math.floor(i / 2) * 0.56, 2.6, 0.44, { fs: 12, round: true }));
T(s, "算法给出的是建议等级，定级结论以审批为准；算法不主动降低等级，降级按《分级手册》附录 E 规定的情形重新定级。", 0.76, 5.68, 5.4, 0.9, { fontSize: 12 });
head2(s, "动态复评的触发条件", 6.5, 2.95, 6.07);
s.addTable([
  [hd("触发事件", { fontSize: 12 }), hd("影响", { fontSize: 12 }), hd("依据", { fontSize: 12 })],
  [td("体量变化、聚合合并、脱敏、加工、安全事件等", { bold: true }), td("规则等级与推断风险均可能变化"), td("《分级手册》附录 E", { color: TITLE })],
  [td("新批次数据开放、字段转为公开", { bold: true }), td("公开基底扩大，其余字段的推断风险可能上升"), td("《开放目录》\n《风评手册》第 138 项", { color: TITLE })],
  [td("披露时点到达", { bold: true }), td("目标数据已披露，相应的上调可以解除"), td("《披露细则》2.7\n《分级手册》附录 E", { color: TITLE })],
  [td("制度修订、历史数据累积", { bold: true }), td("规则库、推断场景与推断能力需更新"), td("《分级手册》四(七)", { color: TITLE })],
], { x: 6.5, y: 3.35, w: 6.07, colW: [2.1, 2.2, 1.77], rowH: [0.36, 0.62, 0.62, 0.62, 0.55], border: BD, objectName: nm("table") });
T(s, "复评由事件触发，从当前已审批的等级出发增量重算，只对发生变化的部分重新验证。", 6.5, 6.25, 6.07, 0.4, { fontSize: 12, bold: true, color: RED });
s.addNotes("阶段四执行南网的审批和固化流程。算法给出的是建议等级，定级结论以审批为准。数据目录建议增加八列，记录上调原因和证据。动态复评由事件触发，右边的表列出了四类触发事件及其依据。");

// ================= 小结：与南网文件的对应 =================
s = content("小结：各步骤与南网文件的对应");
T(s, [{ text: "➢ ", options: { bold: true } }, { text: "体系的每个步骤均有对应的南网条款，并按条款要求执行", options: { bold: true } }], 0.76, 1.2, 11.8, 0.4, { fontSize: 18, valign: "middle" });
const SUM = [
  ["阶段一", "1.1 规则库构建　1.2 数据分类", "《分级手册》四(一)(二)、附录 A", "按业务架构线分类到字段"],
  ["阶段一", "1.3 规则定级　1.4 场景标注", "《分级手册》四(三)、表 4.1~4.4；《披露细则》2.7、5.1", "按判定矩阵定级，就高不就低"],
  ["阶段二", "2.1 推断场景构建", "《披露细则》5.1~5.7；《分级手册》表 4.1~4.3；《开放目录》《注册细则》", "接收方已掌握的信息以文件规定为准"],
  ["阶段二", "2.2 度量　2.3 计算　2.4 验证", "《分级手册》要素“深度”、附录 D、E；《风评手册》第 382、330、351 项", "把聚合推断的原则性要求落实为可计算的方法"],
  ["阶段三", "3.1 等级一致性检查", "《分级手册》表 4.1~4.3 共享开放要求、“从严性”", "消除各角色可见范围内的全部越线组合"],
  ["阶段三", "3.2 等级优化　3.3 复核意见", "《分级手册》“合理性”、级别定义、表 4.4", "避免过度定级；4、5 级只提申报建议"],
  ["阶段四", "4.1 审批　4.2 结果固化", "《分级手册》四(四)(五)", "建议等级经审批后固化到数据目录"],
  ["阶段四", "4.3 动态复评", "《分级手册》四(七)、附录 E；《风评手册》第 138、154 项", "事件触发复评，不主动降级"],
];
const rowsS = [[hd("阶段"), hd("步骤"), hd("南网依据"), hd("遵循方式")]];
SUM.forEach((r) => rowsS.push([td(r[0], { bold: true, align: "center", fill: { color: B0 } }), td(r[1], { bold: true }), td(r[2], { color: TITLE }), td(r[3])]));
s.addTable(rowsS, { x: 0.76, y: 1.72, w: 11.81, colW: [0.95, 2.9, 4.66, 3.3], rowH: 0.47, border: BD, objectName: nm("table") });
dbox(s, 0.76, 6.05, 11.81, 0.72);
T(s, "与现行规定保持一致的处理：等级采用{{南网五级}}；算法{{只提出建议等级}}，定级以审批为准；{{不低于规则等级}}；{{法定公开字段不调整}}；泄露判定线由数据主管部门确定。", 0.95, 6.08, 11.45, 0.66, { fontSize: 13, valign: "middle" });
s.addNotes("最后用一张表做小结：体系的每个步骤都有对应的南网条款，右边一列是遵循的方式。下方列出了与现行规定保持一致的几项处理。");

// ================= 20 提纲 / 21 下一步 / 22 结束 =================
outline(3);
s = content("下一步工作");
[["1、确认体系中的关键口径", "与南网确认敏感目标清单及各自的泄露判定线、接收方角色与可见层级的划分、法定公开字段清单。"],
 ["2、完善推断风险计算与等级优化方法", "完善重训验证流程和等级优化的求解方式，在公开电力数据集上完成对比验证，并形成方法说明文档。"],
 ["3、开展试点应用", "优先选取系统运行域与电力市场披露数据，以发电企业特定信息和运行日次日披露的运行信息为敏感目标开展试点，需协调提供字段级历史样本数据。"]].forEach((it, i) => {
  const y = 1.4 + i * 1.75;
  T(s, it[0], 0.76, y, 11.8, 0.45, { fontSize: 20, bold: true, valign: "middle" });
  T(s, it[1], 1.2, y + 0.55, 11.3, 1.0, { fontSize: 17, lineSpacingMultiple: 1.25 });
});
s = pres.addSlide({ masterName: "COVER" });
T(s, "谢谢大家", 1.2, 2.0, 8, 1.6, { fontSize: 80, bold: true, color: YE, valign: "middle" });

pres.writeFile({ fileName: OUT }).then(async () => {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT)); const tp = "ppt/theme/theme1.xml";
  let xml = await zip.file(tp).async("string"); xml = xml.replace(/<a:ea typeface="[^"]*"\/>/g, '<a:ea typeface="微软雅黑"/>'); zip.file(tp, xml);
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("written", OUT);
});
