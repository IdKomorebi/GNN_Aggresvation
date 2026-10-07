// V6：面向南网汇报；按组内项目汇报模板的风格（蓝色标题 + 细线 + 校徽、➢ 小标题、红色关键词、虚线要点框、分层框架图、编号步骤卡片）
// 运行：export PATH=/data1/duhaocun/envs/node_pptx/bin:$PATH NODE_PATH=/data1/duhaocun/envs/node_pptx_work/node_modules; node V6/scripts/build_ppt.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const V6 = path.resolve(__dirname, "..");
const A = (f) => path.join(V6, "assets", f);
const OUT = path.join(V6, "南网分类分级体系框架_V6.pptx");

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

// ================= 6 提纲 =================
outline(1);

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

// ================= 7 输入、输出与约束 =================
s = content("体系框架：输入、输出与约束");
T(s, [{ text: "➢ ", options: { bold: true } }, { text: "体系的输入、输出与约束条件", options: { bold: true } }], 0.76, 1.2, 11.8, 0.4, { fontSize: 18, valign: "middle" });
panel(s, "输入", 0.76, 1.75, 3.1, 3.95);
[["南网规则文件", "五份文件的条款、表格与示例"], ["数据资产清单", "字段的业务归属、名称与描述"], ["样本数据", "字段级历史样本"], ["敏感目标与泄露判定线", "由数据主管部门确定"]].forEach((it, i) => {
  box(s, [{ text: it[0] + "\n", options: { bold: true, fontSize: 12.5 } }, { text: it[1], options: { fontSize: 10.5, color: MUTE } }], 0.92, 2.25 + i * 0.85, 2.78, 0.72, { fill: "FFFFFF" });
});
arrR(s, 3.95, 3.55, 0.45, 0.36);
panel(s, "四个阶段", 4.5, 1.75, 4.33, 3.95);
SNAME.forEach((t, i) => {
  box(s, SNUM[i], 4.66, 2.25 + i * 0.85, 1.05, 0.72, { fill: SC[i][0], line: SC[i][0], color: "FFFFFF", fs: 12.5, bold: true });
  box(s, t, 5.71, 2.25 + i * 0.85, 2.96, 0.72, { fill: SC[i][1], line: SC[i][0], fs: 13, bold: true });
});
arrR(s, 8.92, 3.55, 0.45, 0.36);
panel(s, "输出", 9.47, 1.75, 3.1, 3.95);
[["分类分级清单", "分类路径、规则等级、建议等级、依据条款"], ["推断证据表", "被推断目标、背景、推断精度、验证方式"], ["等级复核意见", "供审批使用的文字说明"], ["复评记录", "触发事件、等级变动及原因"]].forEach((it, i) => {
  box(s, [{ text: it[0] + "\n", options: { bold: true, fontSize: 12.5 } }, { text: it[1], options: { fontSize: 10.5, color: MUTE } }], 9.63, 2.25 + i * 0.85, 2.78, 0.72, { fill: "FFFFFF" });
});
dbox(s, 0.76, 5.85, 11.81, 0.85, RED);
T(s, "约束", 0.9, 5.88, 0.8, 0.79, { fontSize: 14, bold: true, color: RED, valign: "middle" });
[["建议等级不低于规则等级", "《分类分级手册》四(三)"], ["法定公开字段等级不调整", "《披露细则》第 5 部分"], ["4、5 级只提出申报建议", "《分类分级手册》级别定义"], ["定级结论以审批为准", "《分类分级手册》四(四)"]].forEach((c, i) => {
  T(s, [{ text: c[0], options: { bold: true, fontSize: 12.5, breakLine: true } }, { text: c[1], options: { fontSize: 10.5, color: TITLE } }], 1.7 + i * 2.72, 5.9, 2.65, 0.75, { valign: "middle", align: "center" });
});
basis(s, "输出格式与《分类分级手册》四(五)“结果固化到数据目录”衔接；各项约束的出处见上");
s.addNotes("这一页说明体系的边界。输入是南网规则文件、数据资产清单、样本数据，以及由主管部门确定的敏感目标和泄露判定线。经过四个阶段，输出分类分级清单、推断证据表、复核意见和复评记录。底部是贯穿全程的四条约束。");

// ================= 8 总体结构 =================
s = content("体系框架：总体结构");
T(s, "四个阶段及其步骤", 0.76, 1.2, 6, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle" });
T(s, "对应的南网依据", 9.75, 1.2, 2.82, 0.3, { fontSize: 13, bold: true, color: TITLE, valign: "middle", align: "center" });
const FOCUS = ["执行南网规则，得到每个字段的规则等级", "量化字段在接收方已掌握信息下的推断风险", "在南网约束下确定建议等级", "审批定级，并随变化持续复评"];
const STEPS = [
  ["**1.1 规则库构建**\n解析规则文件", "**1.2 数据分类**\n分类到字段", "**1.3 规则定级**\n等级判定矩阵", "**1.4 场景标注**\n披露层级与时点"],
  ["**2.1 推断场景构建**\n接收方已掌握的信息", "**2.2 推断风险度量**\n字段推断风险 M", "**2.3 推断风险计算**\n关联分析、集合扫描", "**2.4 重训验证**\n确认越线组合"],
  ["**3.1 等级一致性检查**\n可见范围内无越线组合", "**3.2 等级优化**\n上调幅度最小", "**3.3 复核意见生成**\n建议等级 + 推断证据"],
  ["**4.1 审批**", "**4.2 结果固化**\n写入数据目录", "**4.3 动态复评**\n事件触发，增量重算"],
];
const BASIS = [
  "《分类分级手册》四(一)~(三)、表 4.1~4.4\n《披露细则》2.7、5.1",
  "《分类分级手册》要素“深度”、附录 D、E\n《风险评估手册》第 382、330 项\n《披露细则》5.2~5.7",
  "《分类分级手册》“从严性”“合理性”\n表 4.1~4.3 共享开放要求\n重要、核心数据认定要求",
  "《分类分级手册》四(四)(五)(七)、附录 E\n《风险评估手册》第 138、154 项",
];
STEPS.forEach((steps, k) => {
  const y = 1.58 + k * 1.3, h = 1.12, c = SC[k];
  // 左侧阶段块：编号 + 名称 + 要点
  box(s, "", 0.76, y, 2.45, h, { fill: c[0], line: c[0] });
  T(s, [{ text: SNUM[k] + "  ", options: { fontSize: 12, bold: true, color: "FFFFFF" } }, { text: SNAME[k], options: { fontSize: 14.5, bold: true, color: "FFFFFF" } }], 0.88, y + 0.08, 2.25, 0.4, { valign: "middle" });
  T(s, FOCUS[k], 0.88, y + 0.5, 2.25, 0.55, { fontSize: 10.5, color: "FFFFFF", valign: "top" });
  // 步骤
  box(s, "", 3.21, y, 6.4, h, { fill: "FFFFFF", line: c[0] });
  const n = steps.length, gap = 0.26, x0 = 3.33, w = (6.16 - gap * (n - 1)) / n;
  steps.forEach((it, i) => {
    const x = x0 + i * (w + gap);
    box(s, it, x, y + 0.14, w, h - 0.28, { fill: c[1], line: c[0], fs: 10 });
    if (i < n - 1) arrR(s, x + w + 0.03, y + h / 2 - 0.11, gap - 0.06, 0.22, c[0]);
  });
  // 依据
  box(s, "", 9.75, y, 2.82, h, { fill: "F2F2F2", line: "A6A6A6", dash: "dash" });
  T(s, BASIS[k], 9.83, y + 0.04, 2.7, h - 0.08, { fontSize: 9, valign: "middle" });
  if (k < 3) arrD(s, 1.78, y + h + 0.01, 0.4, 0.17, "A6A6A6");
  if (k === 1 || k === 2) box(s, "针对关联推断风险新增", 7.86, y - 0.1, 1.75, 0.2, { fill: RED, line: RED, color: "FFFFFF", fs: 9, bold: true });
});
T(s, "阶段一、阶段四执行南网既有流程；阶段二、阶段三为针对关联推断风险新增的环节，其结果以建议等级和证据的形式进入阶段四。", 0.76, 6.8, 11.81, 0.3, { fontSize: 11.5, bold: true, color: RED, valign: "middle" });
s.addNotes("这是体系的总体结构，共四个阶段、十四个步骤。左侧色块是各阶段的名称和要点，中间是步骤，右侧是对应的南网依据。阶段一和阶段四执行南网已有的流程；阶段二和阶段三是针对关联推断风险新增的环节。后面的关键方法部分按这里的步骤编号逐一展开。");

// ================= 9 提纲 =================
outline(2);

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

// ================= 10 阶段一 (1/2) =================
s = method6("阶段一：基础分类分级（1/2）", "《分类分级手册》四(一)(二)、附录 A");
stepH(s, "1.1 规则库构建", "将南网规则文件解析为结构化规则", 0.76, 1.25, 11.8);
box(s, [{ text: "南网规则文件", options: { bold: true, fontSize: 14, breakLine: true } }, { text: "条款、表格与示例", options: { fontSize: 11 } }], 0.76, 1.95, 2.3, 1.0, { fill: GRAY, line: "A6A6A6" });
arrR(s, 3.14, 2.3, 0.5, 0.32);
box(s, [{ text: "大语言模型 + 检索增强", options: { bold: true, fontSize: 14, breakLine: true } }, { text: "按字段约束抽取并校验", options: { fontSize: 11 } }], 3.72, 1.95, 2.9, 1.0, {});
arrR(s, 6.7, 2.3, 0.5, 0.32);
box(s, [{ text: "结构化规则库", options: { bold: true, fontSize: 14, breakLine: true } }, { text: "可检索、可追溯到条款", options: { fontSize: 11 } }], 7.28, 1.95, 2.3, 1.0, { fill: "BDD7EE" });
exTag(s, 9.95, 1.72);
s.addTable([
  [hdE("规则记录"), hdE("内容")],
  [tdE("数据类别"), tdE("调度运行管理数据")],
  [tdE("影响对象 / 程度"), tdE("公共利益 / 一般影响")],
  [tdE("等级 / 依据"), tdE("3 级 / 《分类分级手册》表 4.1")],
], { x: 9.95, y: 1.98, w: 2.62, colW: [1.12, 1.5], rowH: [0.28, 0.3, 0.3, 0.42], border: BDE, objectName: nm("table") });
stepH(s, "1.2 数据分类", "按企业级业务架构线分类到字段", 0.76, 3.72, 11.8);
["专业域", "专业级流程组", "操作级流程", "业务对象", "数据字段"].forEach((t, i) => chev(s, t, 0.76 + i * 2.33, 4.4, 2.48, 0.6, i === 0, i === 4, 14));
exTag(s, 0.76, 5.25);
s.addTable([
  [tdE("人力资源域", { align: "center" }), tdE("规划与计划管理", { align: "center" }), tdE("人力资源综合计划业务", { align: "center" }), tdE("单位工作计划信息", { align: "center" }), tdE("生效日期、单位、单位类型等", { align: "center" })],
], { x: 0.76, y: 5.55, w: 11.81, colW: [2.3, 2.33, 2.33, 2.33, 2.52], rowH: 0.42, border: BDE, fill: { color: EXF }, objectName: nm("table") });
T(s, "示例取自《分类分级手册》表 A.2", 1.46, 5.25, 6, 0.24, { fontSize: 10.5, color: MUTE, valign: "middle" });
s.addNotes("阶段一的前两步。第一步用大语言模型加检索增强，把南网文件解析成结构化规则库，每条规则都能追溯到条款。第二步按企业级业务架构做线分类，一直分到数据字段。灰色部分是示例。");

// ================= 11 阶段一 (2/2) =================
s = method6("阶段一：基础分类分级（2/2）", "《分类分级手册》四(三)、表 4.4；《披露细则》2.7、5.1");
stepH(s, "1.3 规则定级", "按影响对象与影响程度判定", 0.76, 1.25, 5.7);
const LV = { 1: ["DEEAF6", K], 2: ["BDD7EE", K], 3: ["9DC3E6", K], 4: ["2E75B6", "FFFFFF"], 5: [B2, "FFFFFF"] };
const mat = [["国家安全", 1, 4, 4, 5, 5], ["公共利益", 1, 3, 3, 4, 5], ["经济运行", 1, 2, 3, 4, 5], ["社会秩序", 1, 2, 3, 4, 5], ["其他组织", 1, 2, 2, 3, 3], ["公民个人", 1, 2, 2, 3, 3], ["单位自身", 1, 2, 2, 3, 3]];
const rowsM = [[hd("影响对象", { fontSize: 12 }), ...["无", "轻微", "一般", "严重", "特别严重"].map((h) => hd(h, { fontSize: 12 }))]];
mat.forEach((r) => rowsM.push([td(r[0], { bold: true, align: "center" }), ...r.slice(1).map((v) => td(`${v} 级`, { align: "center", bold: true, fill: { color: LV[v][0] }, color: LV[v][1] }))]));
s.addTable(rowsM, { x: 0.76, y: 1.95, w: 5.6, colW: [1.35, 0.75, 0.83, 0.83, 0.83, 1.01], rowH: 0.44, border: { type: "solid", pt: 0.75, color: "FFFFFF" }, objectName: nm("table") });
T(s, "群体、个体、单位自身三个维度分别判定，取最高者作为{{规则等级}}。", 0.76, 5.65, 5.6, 0.6, { fontSize: 13, valign: "middle" });
stepH(s, "1.4 场景标注", "标注字段的披露属性", 6.75, 1.25, 5.82);
s.addTable([
  [hd("标注项", { fontSize: 12 }), hd("取值", { fontSize: 12 }), hd("依据", { fontSize: 12 })],
  [td("披露层级", { bold: true, align: "center" }), td("公众信息 / 公开信息 / 特定信息 / 内部数据"), td("《披露细则》5.1", { color: TITLE })],
  [td("披露时点", { bold: true, align: "center" }), td("预测类：交易申报前\n出清类：交易日\n运行类：运行日次日"), td("《披露细则》2.7", { color: TITLE })],
  [td("是否法定公开", { bold: true, align: "center" }), td("法定公开字段的等级不作调整"), td("《披露细则》第 5 部分", { color: TITLE })],
  [td("是否敏感目标", { bold: true, align: "center" }), td("特定信息、规则等级 3 级及以上的数据"), td("《分类分级手册》\n表 4.1~4.3", { color: TITLE })],
], { x: 6.75, y: 1.95, w: 5.82, colW: [1.35, 2.62, 1.85], rowH: [0.44, 0.66, 1.0, 0.66, 0.76], border: BD, objectName: nm("table") });
T(s, "上述标注用于阶段二确定各类接收方已掌握的信息。", 6.75, 5.65, 5.82, 0.6, { fontSize: 13, valign: "middle" });
s.addNotes("第三步用南网的判定矩阵得到规则等级。第四步给每个字段标注披露层级、披露时点、是否法定公开，以及是否属于敏感目标，为下一阶段做准备。");

// ================= 12 阶段二 2.1 =================
s = method6("阶段二：推断风险识别（1/3）", "《披露细则》5.1~5.7、2.7；《分类分级手册》表 4.1~4.3；《数据开放目录》说明二(三)；《注册细则》5.10~5.12");
stepH(s, "2.1 推断场景构建", "确定各类接收方已掌握和可能获得的信息", 0.76, 1.25, 11.8);
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

// ================= 13 阶段二 2.2 =================
s = method6("阶段二：推断风险识别（2/3）", "《分类分级手册》分级要素“深度”、分级原则“从严性”");
stepH(s, "2.2 推断风险度量", "以集合推断能力为基础定义字段推断风险", 0.76, 1.25, 11.8);
box(s, "基础量", 0.76, 1.95, 1.0, 0.95, { fill: B2, line: B2, color: "FFFFFF", fs: 13, bold: true });
box(s, "", 1.76, 1.95, 5.74, 0.95, { fill: "FFFFFF", line: B1 });
F(s, "集合推断能力  V_{y}(S)", 1.9, 1.98, 5.5, 0.4, { fontSize: 16, bold: true, color: TITLE });
T(s, "仅用字段集合 S 推断敏感目标 y 所能达到的精度（R²，0~1）", 1.9, 2.4, 5.5, 0.44, { fontSize: 11.5, valign: "middle" });
arrD(s, 3.9, 2.94, 0.44, 0.22);
box(s, "核心\n指标", 0.76, 3.2, 1.0, 1.55, { fill: OR, line: OR, color: "FFFFFF", fs: 13, bold: true });
box(s, "", 1.76, 3.2, 5.74, 1.55, { fill: OR0, line: OR, lw: 1.5 });
F(s, "字段推断风险  M_{i}^{(K)} = max Δ_{i}(B∪T)，T ⊆ H，|T| ≤ K", 1.9, 3.24, 5.5, 0.42, { fontSize: 15.5, bold: true, color: RED });
F(s, "Δ_{i}(B∪T) = V_{y}(B∪T∪{i}) − V_{y}(B∪T)", 1.9, 3.68, 5.5, 0.38, { fontSize: 13.5 });
T(s, "在背景 B∪T 之上再获得字段 i 带来的推断精度提升，对所有不超过 K 个背景字段的组合取最大值", 1.9, 4.08, 5.5, 0.62, { fontSize: 11.5, valign: "middle" });
box(s, "派生量", 0.76, 4.9, 1.0, 0.82, { fill: "7F7F7F", line: "7F7F7F", color: "FFFFFF", fs: 12, bold: true });
box(s, "", 1.76, 4.9, 2.82, 0.82, { fill: "FFFFFF", line: "A6A6A6" });
F(s, "M_{i}^{(0)} = Δ_{i}(B)", 1.88, 4.92, 2.6, 0.4, { fontSize: 13.5, bold: true });
T(s, "单字段风险（K = 0）", 1.88, 5.32, 2.6, 0.36, { fontSize: 11, valign: "middle" });
box(s, "", 4.68, 4.9, 2.82, 0.82, { fill: "FFFFFF", line: "A6A6A6" });
F(s, "Γ_{i}^{(K)} = M_{i}^{(K)} − M_{i}^{(0)}", 4.8, 4.92, 2.6, 0.4, { fontSize: 13.5, bold: true });
T(s, "背景放大量", 4.8, 5.32, 2.6, 0.36, { fontSize: 11, valign: "middle" });
box(s, "判定\n标准", 0.76, 5.87, 1.0, 0.7, { fill: RED, line: RED, color: "FFFFFF", fs: 12, bold: true });
box(s, "", 1.76, 5.87, 5.74, 0.7, { fill: "FFFFFF", line: RED });
T(s, rt("泄露判定线 τ_{y}：推断精度超过 τ_{y} 的字段组合称为**越线组合**"), 1.9, 5.89, 5.5, 0.66, { fontSize: 12, valign: "middle" });
// 示例（弱化）
box(s, "", 7.85, 1.95, 4.72, 4.62, { fill: EXF, line: EXL, dash: "dash" });
exTag(s, 8.0, 2.08);
T(s, "评估字段 A，K = 2，τ = 0.7", 8.72, 2.08, 3.7, 0.24, { fontSize: 11, color: "404040", valign: "middle" });
s.addTable([
  [hdE("背景"), hdE("V(背景)"), hdE("V(背景∪{A})"), hdE("Δ")],
  [tdE("B", { align: "center" }), tdE("0.10", { align: "center" }), tdE("0.45", { align: "center" }), tdE(rt("Δ_{A}(B) = 0.35"), { align: "center" })],
  [tdE("B∪{C}", { align: "center" }), tdE("0.14", { align: "center" }), tdE("0.74", { align: "center" }), tdE(rt("Δ_{A}(B∪C) = 0.60"), { align: "center" })],
  [tdE("B∪{D}", { align: "center" }), tdE("0.12", { align: "center" }), tdE("0.80", { align: "center", bold: true }), tdE(rt("Δ_{A}(B∪D) = 0.68", { bold: true }), { align: "center" })],
], { x: 8.0, y: 2.45, w: 4.42, colW: [0.85, 0.85, 1.17, 1.55], rowH: [0.5, 0.4, 0.4, 0.4], border: BDE, fill: { color: "FFFFFF" }, objectName: nm("table") });
F(s, "M_{A}^{(0)} = Δ_{A}(B) = 0.35", 8.1, 4.35, 4.3, 0.38, { fontSize: 13, color: "404040" });
F(s, "M_{A}^{(2)} = Δ_{A}(B∪D) = 0.68", 8.1, 4.75, 4.3, 0.38, { fontSize: 13, bold: true, color: "404040" });
F(s, "Γ_{A}^{(2)} = 0.68 − 0.35 = 0.33", 8.1, 5.15, 4.3, 0.38, { fontSize: 13, color: "404040" });
T(s, "获得 A 后推断精度由 0.12 升至 0.80，超过判定线，{A, D} 为越线组合", 8.1, 5.65, 4.3, 0.8, { fontSize: 11.5, color: "404040", valign: "middle" });
s.addNotes("这一页给出度量的定义。基础量是集合推断能力V：只用某个字段集合，能把敏感目标推断到多准。核心指标是字段推断风险M：在所有不超过K个背景字段的组合上，再获得这个字段带来的推断精度提升的最大值。单字段风险和背景放大量是派生量。右边灰色部分是一个计算示例。");

// ================= 14 阶段二 2.3 2.4 =================
s = method6("阶段二：推断风险识别（3/3）", "《风险评估手册》第 382、330、351 项；《分类分级手册》附录 E");
stepH(s, "2.3 推断风险计算", "关联分析与集合扫描", 0.76, 1.25, 7.8);
stepH(s, "2.4 重训验证", "确认越线组合", 8.82, 1.25, 3.75);
numCard(s, 1, "关联分析", "计算字段间的线性、秩和非线性关联并融合，构建关联图，用于解释字段关系和排定计算优先级。", 0.76, 1.9, 3.75, 2.05, B1);
numCard(s, 2, "集合扫描", "用通用推断模型估计公开基底之上各字段组合的推断能力，得到各字段的推断风险和候选背景。", 4.79, 1.9, 3.75, 2.05, OR);
numCard(s, 3, "重训验证", "对接收方可见范围内的字段组合逐一重训确认是否越线；发现新的越线组合后重新优化，直至无新增。", 8.82, 1.9, 3.75, 2.05, "BF9000");
arrR(s, 4.53, 2.8, 0.24, 0.26); arrR(s, 8.56, 2.8, 0.24, 0.26);
head2(s, "产出：推断证据表（每个越线组合一行）", 0.76, 4.25, 8);
s.addTable([
  [hd("敏感目标", { fontSize: 11.5 }), hd("目标等级", { fontSize: 11.5 }), hd("字段", { fontSize: 11.5 }), hd("接收方角色", { fontSize: 11.5 }), hd("背景", { fontSize: 11.5 }), hd("获得前精度", { fontSize: 11.5 }), hd("获得后精度", { fontSize: 11.5 }), hd("推断风险", { fontSize: 11.5 }), hd("判定线", { fontSize: 11.5 }), hd("验证方式", { fontSize: 11.5 })],
  [tdE("y", { align: "center", fill: { color: EXF } }), tdE("3 级", { align: "center", fill: { color: EXF } }), tdE("A", { align: "center", fill: { color: EXF } }), tdE("协议接收方", { align: "center", fill: { color: EXF } }), tdE("B∪{D}", { align: "center", fill: { color: EXF } }), tdE("0.12", { align: "center", fill: { color: EXF } }), tdE("0.80", { align: "center", fill: { color: EXF } }), tdE("0.68", { align: "center", fill: { color: EXF } }), tdE("0.7", { align: "center", fill: { color: EXF } }), tdE("重训验证", { align: "center", fill: { color: EXF } })],
], { x: 0.76, y: 4.7, w: 11.81, colW: [1.1, 1.0, 0.8, 1.4, 1.1, 1.25, 1.25, 1.15, 0.95, 1.81], rowH: [0.44, 0.46], border: BD, objectName: nm("table") });
exTag(s, 0.76, 5.72); T(s, "表中一行为示例", 1.46, 5.72, 4, 0.24, { fontSize: 10.5, color: MUTE, valign: "middle" });
s.addNotes("计算分三步。关联分析用来解释字段关系和排优先级。集合扫描用一个通用推断模型估计各字段组合的推断能力。重训验证对可见范围内的组合逐一重训确认，定级依据以验证结果为准。产出是推断证据表，每个越线组合一行。");

// ================= 15 阶段三 3.1 =================
s = method6("阶段三：等级复核与优化（1/3）", "《分类分级手册》表 4.1~4.3（各级数据的共享开放要求）、分级原则“从严性”");
stepH(s, "3.1 等级一致性检查", "各角色可见范围内不存在越线组合", 0.76, 1.25, 11.8);
const L3 = [["字段等级", 0.9, 2.1], ["可见范围\n（可获得该字段的角色）", 3.8, 2.1], ["各角色的背景池", 3.8, 4.0], ["其余字段的推断风险", 0.9, 4.0]];
L3.forEach((b, i) => box(s, b[0], b[1], b[2], 2.3, 0.85, { fs: 12.5, bold: true, fill: i < 2 ? B0 : "BDD7EE" }));
ln(s, 3.22, 2.52, 3.78, 2.52); ln(s, 4.95, 2.97, 4.95, 3.98); ln(s, 3.78, 4.42, 3.22, 4.42); ln(s, 2.05, 3.98, 2.05, 2.97);
T(s, "共享开放要求", 2.9, 3.0, 1.2, 0.25, { fontSize: 10.5, color: MUTE, align: "center" });
T(s, "重新计算", 3.02, 4.9, 0.95, 0.25, { fontSize: 10.5, color: MUTE, align: "center" });
T(s, "字段等级上调后，可获得它的角色减少，其他字段的推断风险随之下降，各字段的等级需要统一考虑。", 0.76, 5.3, 5.75, 1.1, { fontSize: 13, valign: "middle" });
dbox(s, 6.75, 2.0, 5.82, 2.1);
T(s, "对任一接收方角色 r 和任一等级高于其可见层级的敏感目标 y，该角色凭公开基底加上背景池中不超过 K+1 个字段，对 y 的推断精度不超过泄露判定线：", 6.9, 2.04, 5.52, 1.0, { fontSize: 12.5, valign: "middle" });
F(s, "V_{y}(B_{r} ∪ S) ≤ τ_{y}，  ∀ S ⊆ H_{r}，|S| ≤ K+1", 6.9, 3.1, 5.52, 0.5, { fontSize: 17, bold: true, color: TITLE, align: "center" });
s.addTable([
  [hd("检查结果", { fontSize: 12 }), hd("处理", { fontSize: 12 })],
  [td("无越线组合", { bold: true }), td("规则等级即为建议等级")],
  [td("存在越线组合", { bold: true }), td("进入 3.2 等级优化")],
  [td("仅凭公开基底即可越线", { bold: true }), td("提交数据主管部门研判")],
], { x: 6.75, y: 4.35, w: 5.82, colW: [2.6, 3.22], rowH: [0.42, 0.5, 0.5, 0.5], border: BD, objectName: nm("table") });
s.addNotes("南网每一级数据都对应确定的共享开放范围。一个字段的等级上调后，能拿到它的角色就少了，其他字段的推断风险随之下降，所以各字段的等级需要统一考虑。右边是等级一致性条件，以及检查后的三种处理方式。");

// ================= 16 阶段三 3.2 模型 =================
s = method6("阶段三：等级复核与优化（2/3）", "《分类分级手册》分级原则“从严性”“合理性”；级别定义（重要、核心数据须认定）");
stepH(s, "3.2 等级优化", "满足一致性的前提下使上调幅度最小", 0.76, 1.25, 11.8);
T(s, "目标", 0.76, 1.95, 1.0, 0.3, { fontSize: 13.5, bold: true, color: TITLE });
box(s, "", 0.76, 2.3, 5.6, 0.7, { fill: OR0, line: OR, lw: 1.5 });
F(s, "min  Σ_{i} w_{i} · ( ℓ_{i} − ℓ_{i}^{规则} )", 0.76, 2.32, 5.6, 0.66, { fontSize: 18, bold: true, color: RED, align: "center" });
T(s, "约束", 0.76, 3.2, 1.0, 0.3, { fontSize: 13.5, bold: true, color: TITLE });
[["ℓ_{i} ≥ ℓ_{i}^{规则}", "建议等级不低于规则等级"], ["ℓ_{i} = ℓ_{i}^{规则}（法定公开字段）", "法定公开字段不调整"], ["ℓ_{i} ≤ 3", "涉及 4、5 级时只提出申报建议"], ["等级一致性条件成立", "不存在越线组合"]].forEach((c, i) => {
  const y = 3.55 + i * 0.5;
  box(s, "", 0.76, y, 5.6, 0.43, { fill: "FFFFFF", line: B1 });
  F(s, c[0], 0.9, y, 2.75, 0.43, { fontSize: 13 });
  T(s, c[1], 3.7, y, 2.6, 0.43, { fontSize: 11.5, color: MUTE, valign: "middle" });
});
T(s, rt("ℓ_{i} 为字段 i 的建议等级，w_{i} 为每上调一级的代价"), 0.76, 5.7, 5.6, 0.4, { fontSize: 12, valign: "middle" });
// 示例（弱化）
box(s, "", 6.75, 1.95, 5.82, 4.62, { fill: EXF, line: EXL, dash: "dash" });
exTag(s, 6.9, 2.08);
T(s, "目标 y 为 3 级，判定线 0.7", 7.62, 2.08, 4.8, 0.24, { fontSize: 11, color: "404040", valign: "middle" });
const ex = [["P（出清电价）", "1", "法定公开", "1", false], ["A", "1", "{A,B} 0.78  {A,C} 0.74  {A,D} 0.80", "3", true], ["B", "1", "{A,B} 0.78  {B,C,D} 0.72", "1", false], ["C", "1", "{A,C} 0.74  {B,C,D} 0.72", "1", false], ["D", "2", "{A,D} 0.80  {B,C,D} 0.72", "3", true]];
const rowsE = [[hdE("字段"), hdE("规则等级"), hdE("所在越线组合及推断精度"), hdE("建议等级")]];
ex.forEach((r) => rowsE.push([tdE(r[0], { bold: true, align: "center" }), tdE(r[1], { align: "center" }), tdE(r[2], { fontSize: 10.5 }), tdE(r[3], { align: "center", bold: true, color: r[4] ? RED : "404040" })]));
s.addTable(rowsE, { x: 6.9, y: 2.45, w: 5.52, colW: [1.3, 0.9, 2.42, 0.9], rowH: 0.4, border: BDE, fill: { color: "FFFFFF" }, objectName: nm("table") });
box(s, [{ text: "全部上调", options: { bold: true, fontSize: 12, breakLine: true } }, { text: "合计上调 7 级", options: { fontSize: 13 } }], 6.9, 5.1, 2.66, 0.85, { fill: "FFFFFF", line: EXL, color: "404040" });
box(s, [{ text: "优化结果：仅上调 A、D", options: { bold: true, fontSize: 12, breakLine: true } }, { text: "合计上调 3 级", options: { bold: true, fontSize: 13, color: RED } }], 9.76, 5.1, 2.66, 0.85, { fill: "FFFFFF", line: EXL, color: "404040" });
s.addNotes("等级优化的模型：目标是上调幅度最小，约束包括不低于规则等级、法定公开字段不调整、涉及四五级时只提申报建议，以及等级一致性条件成立。右边灰色部分是示例：四个字段都处在越线组合中，全部上调合计七级，优化后只需上调A和D，合计三级。");

// ================= 17 阶段三 3.2 求解 + 3.3 =================
s = method6("阶段三：等级复核与优化（3/3）", "《分类分级手册》四(三)(四)、分级要素“深度”、附录 D、表 4.4");
stepH(s, "3.2 等级优化的求解", "基于强化学习（GRPO）的序贯决策", 0.76, 1.25, 11.8);
const GC = [["状态", "各字段的当前等级、规则等级、推断风险、上调代价", B1], ["动作", "选择一个字段上调一级，或结束", OR], ["环境", "更新各角色的背景池，重新计算推断风险", "BF9000"], ["奖励", "越线风险的减少量减去上调代价", HEAD]];
GC.forEach((g, i) => {
  const x = 0.76 + i * 3.0;
  box(s, g[0], x, 1.95, 0.9, 1.15, { fill: g[2], line: g[2], color: "FFFFFF", fs: 14, bold: true });
  box(s, g[1], x + 0.9, 1.95, 1.72, 1.15, { fill: "FFFFFF", line: g[2], fs: 11.5 });
  if (i < 3) arrR(s, x + 2.66, 2.4, 0.3, 0.26);
});
ln(s, 11.9, 3.12, 11.9, 3.38, { noArrow: true }); ln(s, 11.9, 3.38, 1.2, 3.38, { noArrow: true }); ln(s, 1.2, 3.38, 1.2, 3.12);
T(s, "逐步决策，直至等级一致；3.2 的各项约束通过限制可选动作实现", 3.2, 3.42, 7.0, 0.28, { fontSize: 11.5, color: MUTE, align: "center" });
stepH(s, "3.3 复核意见生成", "说明建议等级的依据，随建议等级提交审批", 0.76, 3.95, 11.8);
["建议等级", "上调原因", "推断证据", "依据条款"].forEach((c, i) => box(s, c, 0.76, 4.62 + i * 0.5, 2.9, 0.43, { fs: 12.5, bold: true }));
box(s, "", 3.95, 4.62, 8.62, 1.93, { fill: EXF, line: EXL, dash: "dash" });
exTag(s, 4.1, 4.74);
T(s, "字段 A 规则等级为 1 级。协议接收方在已掌握出清电价和字段 D 的情况下再获得字段 A，可将 3 级数据 y 的推断精度由 0.12 提高到 0.80，超过泄露判定线 0.7。依据《分类分级手册》分级要素“深度”和附录 D，字段 A 泄露后的影响程度参照 y 评价，按表 4.4 建议复核为 3 级。", 4.1, 5.02, 8.32, 1.48, { fontSize: 12, color: "404040", valign: "middle" });
s.addNotes("求解采用强化学习的GRPO算法，逐步决定上调哪个字段。状态是各字段的等级和推断风险，动作是选一个字段上调一级，环境重新计算风险，奖励是越线风险的减少量减去上调代价。下面是复核意见的内容，右边灰色部分是一个示例。");

// ================= 18 阶段四 =================
s = method6("阶段四：审批固化与动态复评", "《分类分级手册》四(四)(五)(七)、附录 E；《风险评估手册》第 138、146、154 项");
stepH(s, "4.1 审批与 4.2 结果固化", "建议等级经审批后写入数据目录", 0.76, 1.25, 11.8);
box(s, [{ text: "建议等级", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "复核意见与推断证据", options: { fontSize: 11 } }], 0.76, 1.95, 2.3, 0.8, { fill: "FFF2CC", line: "BF9000" });
arrR(s, 3.13, 2.2);
["4.1 审批", "4.2 结果固化到数据目录", "技术管控", "4.3 动态复评"].forEach((t, i) => chev(s, t, 3.65 + i * 2.18, 1.95, 2.33, 0.8, i === 0, i === 3, 13));
T(s, "数据目录增加的列", 0.76, 3.0, 2.0, 0.42, { fontSize: 12.5, bold: true, color: TITLE, valign: "middle" });
["规则等级", "建议等级", "上调原因", "背景", "推断精度", "适用角色与时点", "复评条件", "审批结论"].forEach((c, i) => box(s, c, 2.75 + i * 1.235, 3.0, 1.16, 0.42, { fs: 11, round: true, fill: "FFFFFF" }));
stepH(s, "4.3 动态复评", "由事件触发，从已审批的等级出发增量重算", 0.76, 3.75, 11.8);
s.addTable([
  [hd("触发事件", { fontSize: 12 }), hd("影响", { fontSize: 12 }), hd("依据", { fontSize: 12 })],
  [td("体量变化、聚合合并、脱敏、加工、安全事件等", { bold: true }), td("规则等级与推断风险均可能变化"), td("《分类分级手册》附录 E", { color: TITLE })],
  [td("新批次数据开放、字段转为公开", { bold: true }), td("公开基底扩大，其余字段的推断风险可能上升"), td("《数据开放目录》；《风险评估手册》第 138 项", { color: TITLE })],
  [td("披露时点到达", { bold: true }), td("目标数据已披露，相应的上调可以解除"), td("《披露细则》2.7", { color: TITLE })],
  [td("制度修订、历史数据累积", { bold: true }), td("规则库、推断场景与推断能力需更新"), td("《分类分级手册》四(七)", { color: TITLE })],
], { x: 0.76, y: 4.4, w: 11.81, colW: [4.0, 4.2, 3.61], rowH: [0.4, 0.44, 0.44, 0.44, 0.44], border: BD, objectName: nm("table") });
s.addNotes("阶段四执行南网的审批和固化流程。算法给出的是建议等级，定级结论以审批为准。数据目录增加八列，记录上调原因和证据。动态复评由事件触发，表中列出了四类触发事件及其依据。");

// ================= 小结：与南网文件的对应 =================
s = content("小结：各步骤与南网文件的对应");
T(s, [{ text: "➢ ", options: { bold: true } }, { text: "体系的每个步骤均有对应的南网条款，并按条款要求执行", options: { bold: true } }], 0.76, 1.2, 11.8, 0.4, { fontSize: 18, valign: "middle" });
const SUM = [
  ["阶段一", "1.1 规则库构建　1.2 数据分类", "《分类分级手册》四(一)(二)、附录 A", "按业务架构线分类到字段"],
  ["阶段一", "1.3 规则定级　1.4 场景标注", "《分类分级手册》四(三)、表 4.1~4.4；《披露细则》2.7、5.1", "按判定矩阵定级，就高不就低"],
  ["阶段二", "2.1 推断场景构建", "《披露细则》5.1~5.7；《分类分级手册》表 4.1~4.3；《数据开放目录》《注册细则》", "接收方已掌握的信息以文件规定为准"],
  ["阶段二", "2.2 度量　2.3 计算　2.4 验证", "《分类分级手册》要素“深度”、附录 D、E；《风险评估手册》第 382、330、351 项", "把聚合推断的原则性要求落实为可计算的方法"],
  ["阶段三", "3.1 等级一致性检查", "《分类分级手册》表 4.1~4.3 共享开放要求、“从严性”", "消除各角色可见范围内的全部越线组合"],
  ["阶段三", "3.2 等级优化　3.3 复核意见", "《分类分级手册》“合理性”、级别定义、表 4.4", "避免过度定级；4、5 级只提申报建议"],
  ["阶段四", "4.1 审批　4.2 结果固化", "《分类分级手册》四(四)(五)", "建议等级经审批后固化到数据目录"],
  ["阶段四", "4.3 动态复评", "《分类分级手册》四(七)、附录 E；《风险评估手册》第 138、154 项", "事件触发复评，不主动降级"],
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
