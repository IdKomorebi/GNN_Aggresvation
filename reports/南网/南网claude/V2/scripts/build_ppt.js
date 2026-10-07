// 生成《南网分类分级体系框架》PPT（V2，最小口径）
// 运行：
//   export PATH=/data1/duhaocun/envs/node_pptx/bin:$PATH
//   export NODE_PATH=/data1/duhaocun/envs/node_pptx_work/node_modules
//   node V2/scripts/build_ppt.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const SKILL = "/data1/duhaocun/.claude/skills/synced/3f044006-ec83-4b78-9a15-82fa7fcfeee2_acfc0238-f1ab-4047-92e5-a09a3a29aa4c/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const V2 = path.resolve(__dirname, "..");
const OUT = path.join(V2, "南网分类分级体系框架_V2.pptx");
const FIG = (f) => path.join(V2, "figures", f);

// 主题：深藏青为主色；蓝=南网规则，橙=师兄论文方法，绿=近期研究（与文档配图一致）
const THEME = {
  name: "CSG-Grading",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "1A2238", lt1: "FFFFFF", dk2: "12284C", lt2: "F1F4F8",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A",
    accent4: "DBE8F9", accent5: "FBE1D6", accent6: "D3F0E4",
    hlink: "2A78D6", folHlink: "4A3AA7",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "面向推断风险的电力数据分类分级体系框架";
pres.author = "项目组";
const C = pres.SchemeColor;
const W = 13.33;
const MUTED = "5B6577";
const FOOT = "南网项目 · 数据分类分级体系框架";

// ---------- 版式 ----------
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.5, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 0.9, fontSize: 20, color: C.accent4, align: "left", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 28, bold: true, color: C.text2, align: "left", valign: "middle" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK", background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 28, bold: true, color: C.background1, align: "left", valign: "middle" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: "A9B6CC", margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: "A9B6CC", align: "right" },
});

// ---------- 小工具 ----------
// "M^{(K)}" → 上标 run；"τ_{y}" → 下标 run
function rt(s, base = {}) {
  const out = [];
  const re = /([\^_])\{([^}]*)\}/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ text: s.slice(last, m.index), options: { ...base } });
    out.push({ text: m[2], options: { ...base, superscript: m[1] === "^", subscript: m[1] === "_" } });
    last = re.lastIndex;
  }
  if (last < s.length) out.push({ text: s.slice(last), options: { ...base } });
  return out;
}
let uid = 0;
const nm = (p) => `${p}-${++uid}`;
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { color: fill || C.background2, width: 0 }, objectName: nm("card") });
}
function txt(s, t, x, y, w, h, o = {}) {
  s.addText(typeof t === "string" ? rt(t) : t, { x, y, w, h, isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", align: "left", objectName: nm("text"), ...o });
}
function chip(s, label, x, y, w, color, h = 0.34) {
  s.addText(label, { x, y, w, h, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.17, fill: { color }, line: { color, width: 0 }, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, objectName: nm("chip") });
}
function num(s, n, x, y, color, d = 0.5) {
  s.addText(String(n), { x, y, w: d, h: d, shape: pres.shapes.OVAL, fill: { color }, line: { color, width: 0 }, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, objectName: nm("num") });
}
function arrow(s, x1, y1, x2, y2, color) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color: color || MUTED, width: 1.75, endArrowType: "triangle" }, objectName: nm("arrow") };
  if (x2 < x1) o.flipH = true;
  if (y2 < y1) o.flipV = true;
  s.addShape(pres.shapes.LINE, o);
}
function legend(s, y = 6.55) {
  chip(s, "南网规则", 0.6, y, 1.2, C.accent1);
  chip(s, "师兄论文方法", 1.95, y, 1.6, C.accent2);
  chip(s, "近期研究", 3.7, y, 1.2, C.accent3);
}
const bullets = (arr, o = {}) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1, paraSpaceAfter: 6, ...o } }));
const cell = (t, o = {}) => ({ text: typeof t === "string" ? rt(t) : t, options: { fontSize: 13, color: C.text1, valign: "middle", margin: [0.05, 0.1, 0.05, 0.1], ...o } });
const head = (t, o = {}) => cell(t, { bold: true, color: C.background1, fill: { color: C.text2 }, ...o });
const BORDER = { type: "solid", pt: 0.75, color: "D5DBE5" };

// =====================================================================
// 1 封面
pres.addSection({ title: "背景与问题" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "背景与问题" });
s.addText("面向推断风险的电力数据分类分级体系框架", { placeholder: "title" });
s.addText("以南网分类分级规则为基线，融合关联推断风险识别与动态定级", { placeholder: "body" });
txt(s, "南网项目 · 数据分类分级　|　2026 年 10 月", 0.8, 5.0, 8, 0.4, { fontSize: 14, color: "A9B6CC" });
chip(s, "南网规则", 0.8, 5.75, 1.2, C.accent1);
chip(s, "师兄论文方法", 2.15, 5.75, 1.6, C.accent2);
chip(s, "近期研究", 3.9, 5.75, 1.2, C.accent3);
txt(s, "全文用这三种颜色标注每个环节的来源", 5.3, 5.78, 5, 0.3, { fontSize: 12, color: "A9B6CC" });
s.addNotes("本汇报介绍针对南网项目的数据分类分级体系框架。三种颜色贯穿全文：蓝色是南网已有规则，橙色是师兄论文的方法，绿色是近期研究新增的内容。");

// 2 问题
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景与问题" });
s.addText("现行分级看的是数据本身，还没有看“合起来能推出什么”", { placeholder: "title" });
card(s, 0.6, 1.55, 5.9, 2.85, C.accent4);
chip(s, "现行分级回答的问题", 0.9, 1.8, 2.3, C.accent1);
txt(s, "这份数据本身泄露了，后果有多严重？", 0.9, 2.35, 5.3, 0.9, { fontSize: 22, bold: true, color: C.text2 });
txt(s, "8 个分级要素 → 影响对象 × 影响程度 → 1~5 级，就高不就低", 0.9, 3.45, 5.3, 0.7, { fontSize: 14 });
card(s, 6.83, 1.55, 5.9, 2.85, C.accent5);
chip(s, "还没有回答的问题", 7.13, 1.8, 2.1, C.accent2);
txt(s, "它和接收方已有的数据放在一起，能推出什么？", 7.13, 2.35, 5.3, 0.9, { fontSize: 22, bold: true, color: C.text2 });
txt(s, "两个单看都是 1 级的字段，合起来可能把一份 3 级数据还原出来", 7.13, 3.45, 5.3, 0.7, { fontSize: 14 });
txt(s, "南网文件已经提出这个要求，但没有给出方法和判定标准", 0.6, 4.75, 12.1, 0.4, { fontSize: 16, bold: true, color: C.text2 });
const refs = [
  ["分级原则“从严性”", "多因素时按可能造成的最高影响定级"],
  ["分级要素“深度”", "关联、挖掘、融合对隐含信息的刻画"],
  ["附录 D、E", "挖掘出更敏感的数据应提级；汇聚后须分析能否获得更多信息"],
  ["风险评估第 382 项", "基于已公开数据，尝试能否推断出未公开的关联信息"],
];
refs.forEach((r, i) => {
  const x = 0.6 + i * 3.08;
  card(s, x, 5.3, 2.88, 1.45);
  txt(s, r[0], x + 0.2, 5.42, 2.5, 0.35, { fontSize: 14, bold: true, color: C.accent1 });
  txt(s, r[1], x + 0.2, 5.8, 2.5, 0.85, { fontSize: 12 });
});
s.addNotes("南网现行分级按影响对象和影响程度定级，回答的是数据本身泄露的后果。推断风险是另一个问题：数据发出去以后和接收方已有的数据组合起来能推出什么。南网的分级手册和风险评估手册在四处提到了这一要求，但都没有给出具体方法。");

// 3 例子
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "背景与问题" });
s.addText("一个例子：单看每个字段都安全，合起来能反推机组报价", { placeholder: "title" });
s.addImage({ path: FIG("图3_组合推断示例_PPT用.png"), x: 0.6, y: 1.45, w: 12.13, h: 12.13 * 690 / 2252, altText: "两两关联图与组合推断的对比", objectName: nm("img") });
const stats = [
  ["≈ 0", "每个电价单独对目标的推断能力", C.accent1, 1.2],
  ["0.065 → 0.215", "机组出力的推断增益：单独 → 已知本节点电价时", C.accent2, 2.35],
  ["13% ~ 17%", "只看单字段时能找出的风险字段比例（四个电力数据集）", C.accent3, 1.95],
];
stats.forEach((st, i) => {
  const x = 0.6 + i * 4.11;
  card(s, x, 5.4, 3.9, 1.35);
  txt(s, st[0], x + 0.2, 5.4, st[3], 1.35, { fontSize: 22, bold: true, color: st[2], valign: "middle" });
  txt(s, st[1], x + 0.25 + st[3], 5.4, 3.9 - st[3] - 0.4, 1.35, { fontSize: 12, valign: "middle" });
});
s.addNotes("这是论文中的受控市场算例（RTS-GMLC，目标为机组的私有报价加成，属披露细则中的特定信息）。电价是法定公开信息不能收回，需要管住的是补上最后一步的那个字段。四个电价与报价的相关性都接近零，机组出力单独也只有0.065，但出力和电价合起来就能反推报价。在四个电力数据集上的统计表明，只看单字段只能找出13%到17%的风险字段。");

// 4 设计原则
pres.addSection({ title: "体系总览" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "体系总览" });
s.addText("三条设计原则", { placeholder: "title" });
const pr = [
  ["南网规则是硬约束", "等级表、判定矩阵、审批与固化流程原样使用。算法给出的等级不得低于规则等级；法定公开字段不调整；重要、核心数据只出申报建议。", C.accent1],
  ["在师兄体系上升级", "保留“规则初始分级 → 关联风险识别 → 风险驱动动态定级”的框架，以及规则抽取、关联图、攻击验证和 GRPO。", C.accent2],
  ["风险算准并带证据", "风险量由“两两相关”升级为“字段在已有数据背景下的最坏推断增益”。每一次建议上调都说明：对哪个目标、和谁一起、能推到多准。", C.accent3],
];
pr.forEach((p, i) => {
  const x = 0.6 + i * 4.11;
  card(s, x, 1.6, 3.9, 4.6);
  num(s, i + 1, x + 0.3, 1.9, p[2], 0.6);
  txt(s, p[0], x + 0.3, 2.75, 3.3, 0.9, { fontSize: 19, bold: true, color: C.text2 });
  txt(s, p[1], x + 0.3, 3.8, 3.3, 2.2, { fontSize: 14 });
});
legend(s);
s.addNotes("三条原则：第一，一切以南网文件为基准，不出现冲突；第二，师兄的体系框架保留，在其上升级；第三，近期研究负责把推断风险量化，并让每次上调都有可复核的证据。");

// 5 总览
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "体系总览" });
s.addText("体系总览：四个阶段", { placeholder: "title" });
const stages = [
  ["基础分类分级", ["解析南网规则，建规则库", "按业务架构分类到字段", "矩阵判定规则等级（1~5 级）", "标注披露层级与时点"], [["南网规则", C.accent1, 1.1], ["师兄", C.accent2, 0.8]]],
  ["推断风险识别", ["确定各类接收方已知什么", "计算字段集合的推断能力", "得到字段级风险与见证背景", "对高风险项重训认证"], [["近期研究", C.accent3, 1.1], ["师兄", C.accent2, 0.8]]],
  ["等级复核与优化", ["检查等级是否一致", "GRPO 求上调最少的方案", "输出建议等级和推断证据"], [["师兄 GRPO", C.accent2, 1.25], ["近期研究", C.accent3, 1.1]]],
  ["审批与复评", ["走南网审批流程", "结果固化到数据目录", "事件触发的增量复评"], [["南网规则", C.accent1, 1.1]]],
];
stages.forEach((st, i) => {
  const x = 0.6 + i * 3.1;
  card(s, x, 1.6, 2.83, 4.05);
  num(s, i + 1, x + 0.25, 1.82, C.text2, 0.55);
  txt(s, st[0], x + 0.95, 1.82, 1.8, 0.55, { fontSize: 17, bold: true, color: C.text2, valign: "middle" });
  txt(s, bullets(st[1]), x + 0.25, 2.65, 2.4, 2.2, { fontSize: 14 });
  let cx = x + 0.25;
  st[2].forEach((c) => { chip(s, c[0], cx, 5.1, c[2], c[1]); cx += c[2] + 0.12; });
  if (i < 3) arrow(s, x + 2.86, 3.6, x + 3.07, 3.6, THEME.colors.dk2);
});
card(s, 0.6, 5.9, 12.13, 0.85, C.accent4);
txt(s, [{ text: "硬约束　", options: { bold: true, color: C.text2 } }, { text: "不低于规则等级　·　法定公开字段不调整　·　4、5 级只出申报建议　·　算法输出为建议，定级权在审批" }], 0.85, 5.9, 11.7, 0.85, { fontSize: 14, valign: "middle" });
s.addNotes("体系分四个阶段。第一和第四阶段是南网已有流程；第二阶段是推断风险识别，主要是近期研究；第三阶段用师兄的GRPO做等级优化。底部是贯穿全程的四条硬约束。");

// 6 对照表
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "体系总览" });
s.addText("与师兄原体系的对应：框架保留，两处替换，一处改造", { placeholder: "title" });
const cmp = [
  ["规则理解", "大模型 + 检索增强抽取规则", "语料换为南网手册与细则", "保留", C.background2],
  ["初始定级", "FP-Growth 支持度 → 四级", "南网影响矩阵 → 五级", "按南网替换", C.accent4],
  ["关系发现", "Pearson / Spearman / MIC 风险关联图", "保留，用于解释和先验，不做硬筛选", "保留", C.background2],
  ["推断验证", "DNN；GAT + 物理约束", "保留，作为单字段、结构目标的特例", "保留", C.background2],
  ["风险量", "max MIC（字段两两相关）", "V(S) → M^{(0)}、M^{(K)}、Γ^{(K)}（组合与背景）", "近期研究替换", C.accent6],
  ["等级决策", "GRPO 逐字段选等级，合规为罚项", "GRPO 求上调最少的一致等级，合规为硬约束", "保留并改造", C.accent5],
  ["输出", "算法直接给出等级", "建议等级 + 推断证据 → 审批", "按南网对齐", C.accent4],
];
const rows6 = [[head("环节"), head("师兄原体系"), head("本体系"), head("改动")]];
cmp.forEach((r) => rows6.push([cell(r[0], { bold: true }), cell(r[1]), cell(r[2]), cell(r[3], { bold: true, fill: { color: r[4] }, align: "center" })]));
s.addTable(rows6, { x: 0.6, y: 1.55, w: 12.13, colW: [1.5, 3.9, 4.83, 1.9], rowH: 0.6, border: BORDER, objectName: nm("table") });
txt(s, "师兄已建立“规则初始分级 → 关联风险识别 → 风险驱动动态定级”的完整框架；本体系把初始定级换成南网矩阵，把风险量换成集合推断风险。", 0.6, 6.45, 12.1, 0.45, { fontSize: 12, color: MUTED });
s.addNotes("这张表说明哪些没变、哪些被替换。师兄的规则抽取、关联图、攻击验证都保留；初始定级按南网矩阵替换；风险量由最大MIC换成集合推断风险；GRPO保留，但决策的问题改为求上调最少的一致等级。");

// 7 阶段一
pres.addSection({ title: "四个阶段" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段一：按南网规则完成基础分类分级", { placeholder: "title" });
const st1 = [
  ["建规则库", "大模型 + 检索增强解析分级手册、披露细则等，输出结构化规则", C.accent2],
  ["分类到字段", "专业域 → 流程组 → 操作级流程 → 业务对象 → 字段；合并副本字段", C.accent1],
  ["规则定级", "8 要素 → 影响对象 → 影响程度 → 右侧矩阵，三个维度就高不就低", C.accent1],
  ["场景标注", "披露层级、披露时点、是否法定公开、是否敏感目标", C.accent1],
];
st1.forEach((r, i) => {
  const y = 1.6 + i * 1.22;
  num(s, i + 1, 0.6, y, r[2], 0.5);
  txt(s, r[0], 1.3, y - 0.02, 4.6, 0.4, { fontSize: 17, bold: true, color: C.text2 });
  txt(s, r[1], 1.3, y + 0.4, 4.7, 0.7, { fontSize: 13 });
});
const LV = { 1: ["EEF4FB", C.text1], 2: ["CDE2FB", C.text1], 3: ["86B6EF", C.text1], 4: ["2A78D6", C.background1], 5: ["12284C", C.background1] };
const mat = [["国家安全", 1, 4, 4, 5, 5], ["公共利益", 1, 3, 3, 4, 5], ["经济运行", 1, 2, 3, 4, 5], ["社会秩序", 1, 2, 3, 4, 5], ["其他组织", 1, 2, 2, 3, 3], ["公民个人", 1, 2, 2, 3, 3], ["单位自身", 1, 2, 2, 3, 3]];
const rows7 = [[head("影响对象 ＼ 影响程度", { fontSize: 12 }), ...["无", "轻微", "一般", "严重", "特别严重"].map((h) => head(h, { align: "center", fontSize: 12 }))]];
mat.forEach((r) => rows7.push([cell(r[0], { bold: true, fontSize: 12 }), ...r.slice(1).map((v) => cell(`${v} 级`, { align: "center", fontSize: 12, bold: true, fill: { color: LV[v][0] }, color: LV[v][1] }))]));
s.addTable(rows7, { x: 6.4, y: 1.6, w: 6.33, colW: [1.93, 0.8, 0.85, 0.85, 0.85, 1.05], rowH: 0.47, border: BORDER, objectName: nm("table") });
txt(s, "分级手册表 4.4：一般数据 1~3 级，重要数据 4 级，核心数据 5 级", 6.4, 5.45, 6.3, 0.35, { fontSize: 12, color: MUTED });
card(s, 6.4, 5.85, 6.33, 0.9, C.accent5);
txt(s, "替换师兄原来的“FP-Growth 支持度 → 四级”定级方式；规则库只给建议，结果需人工确认。", 6.6, 5.85, 5.95, 0.9, { fontSize: 13, valign: "middle" });
s.addNotes("阶段一完全按南网手册执行。师兄的大模型加检索增强用于解析南网规则；定级使用南网的影响对象乘影响程度矩阵，替换原来的支持度定级。多做的一步是给每个字段标注披露层级和时点，为下一阶段准备。");

// 8 阶段二①
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段二 ①：先确定“谁、已经知道什么”", { placeholder: "title" });
txt(s, "推断风险取决于接收方手里已有的数据。这一点南网文件已经规定：", 0.6, 1.5, 12.1, 0.4, { fontSize: 15 });
const rows8 = [[head("接收方角色"), head("可见层级"), head("默认已知（公开基底）"), head("还可能拿到（背景池）")],
  [cell("社会公众", { bold: true }), cell("1 级", { align: "center" }), cell("公众信息、无条件开放目录、外部公开数据"), cell("其他 1 级字段")],
  [cell("协议接收方", { bold: true }), cell("≤ 2 级", { align: "center" }), cell("同上 + 向市场成员披露的公开信息"), cell("其他 ≤2 级字段")],
  [cell("特定对象", { bold: true }), cell("≤ 3 级", { align: "center" }), cell("同上 + 获准的特定信息"), cell("其他 ≤3 级字段")]];
s.addTable(rows8, { x: 0.6, y: 2.05, w: 12.13, colW: [2.0, 1.4, 5.0, 3.73], rowH: 0.55, border: BORDER, objectName: nm("table") });
txt(s, "依据：分级手册各级的共享开放口径；披露细则“公众 / 公开 / 特定”三层级。可见层级的划分需南网确认。", 0.6, 4.35, 12.1, 0.35, { fontSize: 12, color: MUTED });
txt(s, "披露时点决定“什么时候已知”", 0.6, 4.9, 6, 0.4, { fontSize: 16, bold: true, color: C.text2 });
const tl = [["D-1 申报前", "预测类信息公开"], ["D 日 交易日", "出清类信息公开"], ["D+1 运行日次日", "运行类信息公开"]];
tl.forEach((t, i) => {
  const x = 0.6 + i * 2.75;
  card(s, x, 5.45, 2.4, 1.2, C.accent6);
  txt(s, t[0], x + 0.15, 5.55, 2.1, 0.4, { fontSize: 15, bold: true, color: C.text2 });
  txt(s, t[1], x + 0.15, 6.0, 2.1, 0.5, { fontSize: 13 });
  if (i < 2) arrow(s, x + 2.43, 6.05, x + 2.72, 6.05, THEME.colors.accent3);
});
card(s, 8.95, 5.45, 3.78, 1.2);
txt(s, "同一字段的风险随时点变化：运行信息在 D+1 公开后，针对它的推断不再构成风险（分级手册附录 E“时效性”）。", 9.1, 5.5, 3.5, 1.1, { fontSize: 12, valign: "middle" });
s.addNotes("推断风险评估的第一步是确定场景。南网分级手册规定了每一级数据可以给谁，披露细则规定了市场信息的三层级和三个披露时点。由此可以得到每类接收方默认已知什么、还可能拿到什么。");

// 9 阶段二②
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段二 ②：把推断风险量化到字段", { placeholder: "title" });
card(s, 0.6, 1.55, 12.13, 0.95, C.accent6);
txt(s, [{ text: "集合推断能力 V(S)　", options: { bold: true, color: C.text2 } }, { text: "只用字段集合 S 训练的最强攻击模型，对敏感目标的推断精度（R²）。师兄的 DNN 验证是它在单字段时的特例。" }], 0.85, 1.55, 11.7, 0.95, { fontSize: 15, valign: "middle" });
const qs = [
  ["M^{(0)}", "单独风险", "这个字段自己能让目标多被推出多少？", "对应师兄原来的单字段口径"],
  ["M^{(K)}", "最坏背景下的风险", "在别人已有的数据背景下，它最多能多推出多少？", "取最大值，对应“从严性”"],
  ["Γ^{(K)}", "背景放大量", "它的风险有多少是被背景放大的？", "等于 M^{(K)} − M^{(0)}"],
];
qs.forEach((q, i) => {
  const x = 0.6 + i * 4.11;
  card(s, x, 2.75, 3.9, 2.55);
  txt(s, rt(q[0]), x + 0.25, 2.85, 1.5, 0.75, { fontSize: 30, bold: true, color: C.accent3 });
  txt(s, q[1], x + 1.75, 3.02, 2.05, 0.45, { fontSize: 15, bold: true, color: C.text2 });
  txt(s, q[2], x + 0.25, 3.7, 3.4, 0.85, { fontSize: 14 });
  txt(s, rt(q[3]), x + 0.25, 4.65, 3.4, 0.5, { fontSize: 12, color: MUTED });
});
txt(s, "同时记录见证背景（和哪几个字段一起）和可达精度（加上它以后目标能被推到多准）。", 0.6, 5.42, 12.1, 0.35, { fontSize: 13 });
const calc = [["通用模型扫描", "48~64 毫秒/组合；逐个重训需 14~56 秒", C.accent3], ["重训认证", "最强攻击模型、全量数据重训确认", C.accent3], ["风险关联图", "师兄方法并联：解释、先验、副本提示", C.accent2]];
calc.forEach((c, i) => {
  const x = 0.6 + i * 4.11;
  chip(s, c[0], x, 6.0, 1.6, c[2]);
  txt(s, c[1], x + 1.72, 5.87, 2.2, 0.6, { fontSize: 12, valign: "middle" });
});
s.addNotes("风险量化的核心是集合推断能力V，即只用某个字段集合能把目标推到多准。在此基础上得到三个字段级的量：单独风险、最坏背景下的风险、背景放大量。计算上用一个通用模型快速扫描所有组合，再对高风险项重训认证。师兄的关联图并联使用。");

// 10 阶段三①
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段三 ①：等级要一起定，因为等级决定谁能拿到数据", { placeholder: "title" });
const loop = [["字段等级", 0.9, 1.9, C.accent4], ["可见范围\n哪些角色拿得到", 4.2, 1.9, C.accent4], ["各角色的背景池", 4.2, 4.6, C.accent6], ["其余字段的\n推断风险", 0.9, 4.6, C.accent6]];
loop.forEach((b) => {
  s.addText(b[0], { x: b[1], y: b[2], w: 2.4, h: 1.1, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, fill: { color: b[3] }, line: { color: b[3], width: 0 }, fontSize: 15, bold: true, color: C.text2, align: "center", valign: "middle", objectName: nm("loop") });
});
arrow(s, 3.35, 2.45, 4.15, 2.45); arrow(s, 5.4, 3.05, 5.4, 4.55); arrow(s, 4.15, 5.15, 3.35, 5.15); arrow(s, 2.1, 4.55, 2.1, 3.05);
txt(s, "各级的共享开放口径", 3.0, 1.55, 1.6, 0.3, { fontSize: 11, color: MUTED, align: "center" });
txt(s, "上调的字段\n退出更宽的池", 5.55, 3.5, 1.3, 0.6, { fontSize: 11, color: MUTED });
txt(s, "重算", 3.4, 5.25, 0.8, 0.3, { fontSize: 11, color: MUTED, align: "center" });
txt(s, "是否仍需上调", 0.75, 3.65, 1.3, 0.3, { fontSize: 11, color: MUTED, align: "right" });
card(s, 7.2, 1.55, 5.53, 2.35);
txt(s, "一致性条件", 7.45, 1.7, 5, 0.4, { fontSize: 17, bold: true, color: C.text2 });
txt(s, "任何角色凭已知数据，再加上自己可见字段中的少数几个（不超过 3 个），都不能把高于其权限的敏感目标推断到泄露线以上。", 7.45, 2.15, 5.05, 1.2, { fontSize: 14 });
txt(s, "泄露线由该目标的数据主管部门给出，建议初值 0.7", 7.45, 3.4, 5.05, 0.35, { fontSize: 12, color: MUTED });
card(s, 7.2, 4.1, 5.53, 2.6, C.accent5);
txt(s, "优化目标（最小口径）", 7.45, 4.25, 5, 0.4, { fontSize: 17, bold: true, color: C.text2 });
txt(s, "在满足一致性的前提下，使上调的总量最少。", 7.45, 4.72, 5.05, 0.5, { fontSize: 14 });
chip(s, "从严性", 7.45, 5.4, 1.0, C.accent1); txt(s, "所有越线的组合都要消除", 8.6, 5.4, 3.9, 0.34, { fontSize: 13, valign: "middle" });
chip(s, "合理性", 7.45, 5.9, 1.0, C.accent1); txt(s, "结合保护成本，避免过度定级", 8.6, 5.9, 3.9, 0.34, { fontSize: 13, valign: "middle" });
txt(s, "两条都来自分级手册的分级原则", 7.45, 6.32, 5, 0.3, { fontSize: 11, color: MUTED });
s.addNotes("这一页是体系的关键。南网每一级都绑定了共享开放口径，字段被上调后，更宽的角色就拿不到它，其他字段的风险随之下降。所以上调谁、调几级是相互牵制的。优化目标来自南网的两条分级原则：从严性要求消除所有越线组合，合理性要求避免过度定级。");

// 11 阶段三② 示例
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段三 ②：示例——只调必要的字段", { placeholder: "title" });
txt(s, "示意数字，非实测。目标 y 为 3 级，泄露线 0.7；公众可见 1 级字段，协议接收方可见 ≤2 级字段；P 为法定公开的出清电价。", 0.6, 1.45, 12.1, 0.4, { fontSize: 12, color: MUTED });
const ex = [["P 出清电价", "1", "—", "在公开基底内", "1（锁定）", false], ["A", "1", "0.35 → 0.68", "{A,B} 0.78  {A,C} 0.74  {A,D} 0.80", "3", true], ["B", "1", "0.05 → 0.42", "{A,B} 0.78  {B,C,D} 0.72", "1", false], ["C", "1", "0.04 → 0.42", "{A,C} 0.74  {B,C,D} 0.72", "1", false], ["D", "2", "0.02 → 0.37", "{A,D} 0.80  {B,C,D} 0.72", "3", true]];
const rows11 = [[head("字段"), head("规则等级", { align: "center" }), head(rt("M^{(0)} → M^{(2)}", { bold: true }), { align: "center" }), head("参与的越线组合及其推断精度"), head("建议等级", { align: "center" })]];
ex.forEach((r) => rows11.push([cell(r[0], { bold: true }), cell(r[1], { align: "center" }), cell(r[2], { align: "center" }), cell(r[3], { fontSize: 12 }), cell(r[4], { align: "center", bold: true, fill: { color: r[5] ? C.accent5 : C.background1 } })]));
s.addTable(rows11, { x: 0.6, y: 1.95, w: 7.9, colW: [1.35, 0.95, 1.4, 3.0, 1.2], rowH: 0.52, border: BORDER, objectName: nm("table") });
const steps = [["先调风险最大的 A", "A 到 2 级后公众已拿不到，但协议接收方仍可用 {A,B}，继续调到 3 级"], ["重算", "协议接收方手里还剩 {B,C,D} 越线；调 D 一级即可，比调 B 或 C 两级代价小"], ["再重算", "各角色可见范围内已无越线组合，结束"]];
steps.forEach((st, i) => {
  const y = 1.95 + i * 1.08;
  num(s, i + 1, 8.85, y, C.accent2, 0.45);
  txt(s, st[0], 9.45, y - 0.02, 3.3, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  txt(s, st[1], 9.45, y + 0.33, 3.3, 0.7, { fontSize: 12 });
});
card(s, 0.6, 5.3, 5.9, 1.45);
txt(s, "7 级次", 0.85, 5.3, 2.2, 1.45, { fontSize: 32, bold: true, color: MUTED, valign: "middle" });
txt(s, "若凡参与越线组合的字段都取目标级别：A、B、C、D 全部上调", 3.0, 5.3, 3.3, 1.45, { fontSize: 13, valign: "middle" });
card(s, 6.83, 5.3, 5.9, 1.45, C.accent5);
txt(s, "3 级次", 7.08, 5.3, 2.2, 1.45, { fontSize: 32, bold: true, color: C.accent2, valign: "middle" });
txt(s, "最小口径：只上调 A、D，其余组合自然失效，B、C 保持 1 级", 9.23, 5.3, 3.3, 1.45, { fontSize: 13, valign: "middle" });
s.addNotes("用一个示意例子说明。四个字段都参与了越线组合，如果全部上调到目标级别要调7级次；按最小口径，先调风险最大的A，重算后只需再调D，一共3级次，B和C保持不变。注意表中是示意数字。");

// 12 阶段三③ GRPO
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段三 ③：用师兄的 GRPO 求解，南网规则作硬约束", { placeholder: "title" });
const g = [["状态", "每个字段的当前等级、规则等级、M^{(0)}、M^{(K)}、Γ^{(K)}、上调代价；剩余越线数", C.accent6], ["动作", "把某个字段上调一级，或结束", C.accent5], ["环境", "更新可见范围与背景池，重算受影响组合的风险", C.accent6], ["奖励", "越线风险的减少 − 上调代价；达到一致时给终止奖励", C.accent5]];
g.forEach((b, i) => {
  const x = 0.6 + i * 3.1;
  card(s, x, 1.6, 2.83, 2.2, b[2]);
  txt(s, b[0], x + 0.2, 1.72, 2.4, 0.4, { fontSize: 17, bold: true, color: C.text2 });
  txt(s, rt(b[1]), x + 0.2, 2.2, 2.45, 1.5, { fontSize: 13 });
  if (i < 3) arrow(s, x + 2.86, 2.7, x + 3.07, 2.7, THEME.colors.dk2);
});
card(s, 0.6, 4.05, 5.9, 2.7, C.accent4);
txt(s, "硬约束（不进奖励，直接限制可选动作）", 0.85, 4.18, 5.4, 0.4, { fontSize: 15, bold: true, color: C.text2 });
txt(s, bullets(["等级不低于南网规则等级", "法定公开字段不可调整", "算法最高给到 3 级；涉及重要、核心数据只出申报建议", "上调不超过所涉目标的级别"]), 0.85, 4.65, 5.4, 2.0, { fontSize: 13 });
card(s, 6.83, 4.05, 5.9, 2.7);
txt(s, "相对师兄原设定的变化", 7.08, 4.18, 5.4, 0.4, { fontSize: 15, bold: true, color: C.text2 });
txt(s, bullets(["风险输入：max MIC 换成 M^{(K)}".replace("M^{(K)}", "M(K)"), "状态会随动作变化，成为真正的序列决策", "合规由罚项改为硬约束；四级改为五级", "算法本身（组内相对优势、KL 约束、熵正则）不变"]), 7.08, 4.65, 5.4, 1.55, { fontSize: 13 });
txt(s, "对比基线：贪心（每次上调“消除风险 ÷ 代价”最大的字段）", 7.08, 6.25, 5.4, 0.35, { fontSize: 12, color: MUTED });
s.addNotes("求解沿用师兄的GRPO算法。状态里的风险量换成了近期研究的M，动作是把某个字段上调一级，环境会重算风险，所以状态随动作变化。南网规则全部作为硬约束。对比基线是贪心算法。");

// 13 阶段四
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "四个阶段" });
s.addText("阶段四：审批固化，事件触发复评", { placeholder: "title" });
card(s, 0.6, 1.6, 5.9, 5.1);
txt(s, "输出进入南网流程", 0.85, 1.75, 5.4, 0.4, { fontSize: 17, bold: true, color: C.text2 });
txt(s, "建议等级 + 推断证据 → 审批 → 固化到数据目录", 0.85, 2.2, 5.4, 0.4, { fontSize: 14 });
txt(s, "数据目录建议新增的列", 0.85, 2.85, 5.4, 0.35, { fontSize: 14, bold: true, color: C.text2 });
const cols = ["规则等级", "建议等级", "上调原因", "见证背景", "可达精度", "适用角色与时点", "复评条件", "审批结论"];
cols.forEach((c, i) => {
  const x = 0.85 + (i % 2) * 2.75, y = 3.3 + Math.floor(i / 2) * 0.5;
  chip(s, c, x, y, 2.55, C.accent1, 0.38);
});
txt(s, "证据同时可用于风险评估手册第 382、330、351、239 项的评估意见", 0.85, 5.55, 5.4, 0.9, { fontSize: 12, color: MUTED });
txt(s, "什么时候复评", 6.83, 1.6, 5.9, 0.4, { fontSize: 17, bold: true, color: C.text2 });
const trig = [["分级手册附录 E 的场景", "体量变化、聚合合并、时效变化、脱敏、加工、安全事件"], ["新批次开放或字段转为公开", "已知数据变多，其余字段的风险可能上升"], ["披露时点到达", "目标本身已公开，相应的上调可以解除"], ["细则修订、历史数据累积", "规则库、角色划分和推断能力都要更新"]];
trig.forEach((t, i) => {
  const y = 2.2 + i * 1.02;
  num(s, i + 1, 6.83, y, C.accent1, 0.45);
  txt(s, t[0], 7.45, y - 0.03, 5.2, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  txt(s, t[1], 7.45, y + 0.33, 5.2, 0.5, { fontSize: 12 });
});
card(s, 6.83, 6.25, 5.9, 0.5, C.accent6);
txt(s, "复评从当前已审批的等级出发增量重算；算法不主动降级", 7.0, 6.25, 5.6, 0.5, { fontSize: 12, valign: "middle" });
s.addNotes("算法输出的是建议等级和证据，定级仍走南网审批并固化到数据目录。动态更新是事件触发的，不随运行工况逐时变化。除了分级手册附录E的场景，新增了公开范围变化和披露时点到达两类触发。");

// 14 对应表
pres.addSection({ title: "对应与下一步" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "对应与下一步" });
s.addText("每一步都对应南网条款", { placeholder: "title" });
const map = [
  ["分类、规则定级", "分级手册 四(一)~(三)，表 4.1~4.4，附录 A~C", "执行"],
  ["场景标注（角色、时点）", "披露细则 2.7、5.1~5.7；开放目录说明；注册细则", "引用"],
  ["推断风险识别", "分级要素“深度”；附录 D、E；风险评估第 382、330、351、239 项", "把原则变成可计算的方法"],
  ["取最坏背景 / 上调最少", "分级原则“从严性” / “合理性”", "执行"],
  ["等级与可见范围", "表 4.1~4.3 各级共享开放口径", "引用"],
  ["4、5 级只出申报建议", "重要数据、核心数据须按主管部门要求识别申报", "执行"],
  ["审批、固化、复评", "分级手册 四(四)(五)(七)，附录 E", "执行并具体化"],
];
const rows14 = [[head("体系步骤"), head("南网条款"), head("关系")]];
map.forEach((r) => rows14.push([cell(r[0], { bold: true }), cell(r[1]), cell(r[2])]));
s.addTable(rows14, { x: 0.6, y: 1.55, w: 12.13, colW: [3.0, 6.4, 2.73], rowH: 0.56, border: BORDER, objectName: nm("table") });
txt(s, "已处理的潜在冲突：四级改五级；算法不直接定级；不降低规则等级；不上调法定公开字段；等级不随运行工况逐时变化；泄露线由主管部门定。", 0.6, 6.2, 12.1, 0.7, { fontSize: 12, color: MUTED });
s.addNotes("这张表逐步列出体系与南网条款的对应关系。下方列出了已经处理的潜在冲突点。");

// 15 现状与下一步
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "对应与下一步" });
s.addText("现状与下一步", { placeholder: "title" });
const nx = [
  ["已完成", ["南网 5 组文件梳理", "师兄论文体系梳理", "体系框架设计", "推断风险量化方法（已在四个公开电力数据集验证）"], C.accent3],
  ["进行中", ["等级优化的验证实验：最小口径与全部上调的对比", "贪心与 GRPO 的对比", "用估计值做优化后的残余风险检查"], C.accent2],
  ["需南网确认", ["敏感目标清单与各自的泄露线", "接收方角色与可见层级", "法定公开字段清单", "试点范围与样本数据"], C.accent1],
];
nx.forEach((c, i) => {
  const x = 0.6 + i * 4.11;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 3.9, h: 4.0, rectRadius: 0.08, fill: { color: "1D3763" }, line: { color: "1D3763", width: 0 }, objectName: nm("card") });
  chip(s, c[0], x + 0.3, 1.9, 1.5, c[2], 0.4);
  txt(s, bullets(c[1], { paraSpaceAfter: 10 }), x + 0.3, 2.6, 3.35, 2.8, { fontSize: 16, color: C.background1 });
});
txt(s, "说明：研究数字来自公开或合成电力数据，尚未在南网数据上测试；等级优化为本框架新提出的做法，验证实验正在进行。", 0.6, 5.95, 12.1, 0.6, { fontSize: 13, color: "A9B6CC" });
s.addNotes("目前框架设计已完成，推断风险量化方法已在四个公开电力数据集上验证。等级优化是本框架新提出的做法，验证实验正在进行。需要南网确认四件事：敏感目标与泄露线、角色与可见层级、法定公开字段清单、试点范围。");

// ---------- 输出 ----------
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  // 给主题补东亚字体，避免中文回落到宋体
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT));
  const tp = "ppt/theme/theme1.xml";
  let xml = await zip.file(tp).async("string");
  xml = xml.replace(/<a:ea typeface="[^"]*"\/>/g, '<a:ea typeface="Microsoft YaHei"/>');
  zip.file(tp, xml);
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("written", OUT);
})();
