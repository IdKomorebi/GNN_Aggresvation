# 95 号（caiso）工作日志 — 第二数据集验证

> 目的：在 CAISO 2025 数据上复刻 91→94 号收束的核心结论，补上 PAPER_STORYLINE
> 诚实边界第 4 条（"单数据集"）。任务由 Claude 自主执行，本文件持续追加，防中断丢失。

## 状态：✅ 全部完成（2026-07-27）——结论定稿见 CHANGELOG.md，五幕全部复刻成立

## 已确认事实（2026-07-26）

### 数据
- 第一数据集：PJM RTO 2025 小时数据，72 列 × 8760 行
  （根目录 `pjm_rto_hourly_2025_aligned_processed_one_header.csv`）。
  实验设定：44 可见字段 + 12 机密字段（94 号 base.yaml `fields.confidential`），
  丢 11 列辅助 + 常量列。
- 第二数据集：CAISO 2025 小时数据，**307 列 × 8760 行**，UTC 对齐，
  `data/caiso/caiso_extracted/caiso_2025_hourly_augmented/caiso_2025_hourly_aligned_augmented.csv.gz`。
  163 原始字段 + 142 增补字段，带逐列字典（column_dictionary.csv）与官方来源清单。
  列结构（初看）：actual_load / dam_load_forecast 按 ~34 个 BA 展开、
  renewables 按 NP15/SP15/ZP26 × solar/wind、DA/RT15 LMP 按 hub × 成分
  (energy/congestion/loss/ghg/lmp)、AS requirement 多区域多产品。
- **用户顾虑（属实）**：列太多、辅助数据太多 → 需要字段收缩设计（307 → ~44+12），
  并警惕 LMP=energy+congestion+loss+ghg 的精确恒等式造成可见字段完美共线。

### 算力
- 4×RTX 4090（24GB）。GPU 0 被他人占用（52%），**用 GPU 1、2**（用户指定两张卡）。

### 复刻目标（按 94 号 PAPER_STORYLINE 五幕）
核心待验证主张（PJM 上已认证，CAISO 上待复刻）：
1. 全摊销 oracle 的误差集中在最强协同档（尾部失明）
2. 冻结特征 + 闭式岭读出（L0）大幅修复尾部；L1（闭式头进训练回路）更优
3. 手工字典高阶报假（认证塌陷），L1 自选 top 认证为真
4. 真值版衰减律：阶数↑ 强度温和衰减、密度陡降
5. 跨 seed 秩一致性过滤提升 top 精确率
6. 协同=共线放大（synergy-is-collinearity 闭式 R² 机制，2026-07-26 新发现）

### 方法论红线（必须遵守）
- 重训真值只做评估，**不得进查询/路由/校准路径**（用户两次否决过）
- 认证协议用 88 号修正后的早停版本
- 置换检验对差分假阳性失明（93 号）→ 认证只认重训

## Phase 0 完成（2026-07-26）：可行性判定 = **调整后适合，继续**

调查报告全文存 `investigation/`（4 份：pipeline94 / oracleChain / caisoAudit / certProtocol）。要点：

**CAISO 体检**：8760 行严格完整连续；305 数值列分 11 组；47 全零列 + 10 近零列 +
1 列缺失 98.9% 必须剔除；有效独立维度 ~233。精确恒等式 6 类（LMP=energy+cong+loss+ghg
恒等、energy 跨 hub 复制、负荷/计划/AS 加和恒等）。12 个 PJM 机密全部找到 CAISO 载体，
但 total_gen/interchange 只有日前计划口径（ENE_SLRS），无 RT 实测——写进 limitation。
体检 agent 推荐方案 A（44 可见+12 机密，逐项镜像 PJM），备选 B（WEIM 空间图版）/C（价格全息版）。

**管线复刻清单**（细节见 investigation/pipeline94.md §5）：
- 代码依赖：69 号 src/（prepare_data/MLPOracle/sample_mask，nG/nC 均动态计算，无硬编码 44/12）
- 必改：base.yaml 整段重写；8 个脚本的 csv_path 硬编码覆盖行；paired_fdr.py 的 CKPT 常量（无 --ckpt 参数）
- 固定种子全家桶：切分 42 / SPLIT_SEED=880725 / VAL_SEED=20260724 / 认证早停 20260725 / PICK_SEED=930726 / L1 val 999、掩码 1234+seed
- 坑：certify 断点续跑 glob 不区分 source/tag，换来源前要清空或改 glob
- 多卡 = 同脚本多进程 + CUDA_VISIBLE_DEVICES 分片（代码内固定 cuda）
- 环境：conda env Pytorch310_codex

**认证协议**（88 号修正版）：专用 DNN(128,128,dropout0.15)、Adam 1e-3/5e-4、
EPOCHS400/PATIENCE120、早停用训练集内部 15% val（seed 20260725）、测试集不参与；
syn_true_audit>0.10=真；必带随机对照组 + Mann-Whitney；~160-186s/集合。

**耗时基准**（PJM 实测）：L1 训练 45-234s；o5 全扫 40 GPU·min；FDR o3 ~10min；
认证 top-21 ≈63 GPU·min。

## Phase 1 完成：字段设计定稿 = 方案 A3（三轮泄露审计迭代）

审计脚本 scripts/audit_fields.py，全记录见 fields_design.md：
- A0→A1：RT energy 可见导致 rt_cong_sp15 R²=0.9964（恒等式残差只剩 loss）→ 移出，降至 0.788
- A1→A2：3 个 TAC 实际负荷可见导致机密 ca_iso_tac R²=0.99998（加和恒等式）
  → 换成 WEIM 外部 BA pace/nevp/azps，降至 0.943
- A2→A3：sr/nr AS 清算价整列为零被运行时删常量列（nG=42，冒烟训练暴露）
  → 换 regulation mileage up/down minimum，nG=44 ✅
- A3 最终：最大线性 R²=0.9894 < PJM 最大 0.9984，轮廓 0.24–0.99 与 PJM 形态一致

数据定稿：data/Processed/caiso_2025_hourly_cleaned.csv（56 列，8722 完整行，
运行时 train 6105 / test 2617 / search~1308 / audit~1309），metadata.json 带 SHA-256。

冒烟结果（A2 版数据，仅证管线通）：uniform oracle 满输入 12-conf R²=0.780
（348ep/42s）；L1 aug8 seed0 val=0.342（119ep/26s）。

脚本移植清单：train_l1 / scan_order5 / paired_fdr(+--ckpt 参数) /
certify_highorder(--source 必填 + 断点续跑 glob 修 tag) / eval_l1_order2 / eval_l1_order3 /
analyze_certify / train_oracle(新) / build_truth_o2(新) / build_truth_o3(新,400 三元组
seed950726) / finalize_truth(新)；src/featridge.py 增 full 字典（poly2+3..m 阶乘积）。

## Phase 2 完成：模型训练（A3 正式版，2026-07-26 20:36）
- oracle_uniform_seed0：nG=44 nC=12，ep400，val 0.3395，满输入 12-conf R²=0.7809（55s）
- oracle_l1_aug8_seed{0,1,2}：val 0.3734 / 0.3794 / 0.4095（38/23/31s，122/105/127ep）
- 注：期间发现 A2 有两列全零清算价被删常量列（nG=42），已修为 A3 并重训（见 Phase 1）

## Phase 3 进行中：真值 + 扫描（2026-07-26 21:43 启动）
- 4×build_truth_o2 worker（GPU1: shard0,1；GPU2: shard2,3；990 集合，nohup，
  日志 outputs/truth_o2_shard*.log，进度看 outputs/events.jsonl 的 TRUTH2 条目）
- scans_gpu1.sh（GPU1）：L1s0 缓存→o4 全扫→o5 shard0/2→full o4→full o5 shard0/2
- scans_gpu2.sh（GPU2）：L1s1/s2 o4 全扫→o5 shard1/2→full o5 shard1/2→FDR o3→
  s1/s2 的 subs_o4 缓存
- 之后：finalize_truth → eval_l1_order2/3 → 认证（L1top/fulltop/ctrl，o4+o5）→
  o3 FDR-top 认证 → ensemble filter → 衰减律
- 全零清算价教训：任何"选列"清单必须先对 column_validation_stats 或实测 std 过一遍

### 扫描结果速报（2026-07-26 22:38，全部 audit 口径）
| 扫描 | max_syn | n>0.2 | 用时 |
|---|---|---|---|
| o4 L1 seed0/1/2 | 0.1301 / 0.1311 / 0.1321 | 0/0/0 | ~500s each |
| o5 L1 seed0（两分片） | 0.0855 / 0.0782 | 0 | ~1950s each |
| o4 full 字典 | 0.1264 | 0 | 37s |
| o5 full 字典 shard0 | 0.1059 | 0 | 254s |

★早期信号（跨数据集差异）：**CAISO 上 full 字典没有复现 PJM 的"假强五阶洪水"**
（PJM full o5 曾报 441 个 syn>0.2、max 0.514；CAISO full o5 max 仅 0.106）。
o4 的 L1 max_syn 三个 seed 高度一致（0.130-0.132）。初步解读：CAISO 的高阶协同
整体弱于 PJM（待认证定论）；full 报假的强度可能与数据集共线结构有关。
第三幕复刻形态要变：不再是"full 报假 vs L1 报真"，而是"两者都报弱 + 认证检验
L1 对自身 top 的无偏性 + 衰减律绝对水平对比"。

### FDR order-3 结果（2026-07-26 22:55，cat3 + L0 uniform oracle，776s）
- **τ=0.10 过 FDR：122/13,244 = 0.92%**（PJM：66/13,244 = 0.50%）；τ=0.15：30 个
- 干净名单 syn_audit 范围 0.141–0.374；**统计口径 o3 最强 = 0.374**（PJM 0.461）
- 通过名单被 conf=1（发电计划总量）主导，字段组合 = DA LMP hub 价(20/21) +
  AS spin/nonspin 需求下限(36/37) + 风光(7/9/11/13/17)——物理可解释
- ⚠数值观察：79/13,244 行 syn_audit 爆炸（15–125，闭式解在 audit 极端样本上外推
  失稳；CAISO 价格尖峰重尾比 PJM 极端）。**FDR 自我保护验证：0 个病态行过 FDR**
  （se 同步爆炸→t 变小）。但"硬阈值计数"含污染，报告时只用 FDR 计数。
  36/37 是完全重复列对，但 φ 的 mask 位不同→v 不同，属预期非 bug。
- fdr_o3_top_s0of1.parquet 已生成（122 个，供认证）

### 认证计划（certs_shard.sh，truth_o2 完成后启动，2 分片×2 GPU）
1. o4 L1-top20+ctrl20(_l1top) 2. o5 L1-top21+ctrl20(_l1top) 3. o3 FDR-top10+ctrl10(_fdrtop)
4. o4 full-top20(_fulltop) 5. o5 full-top20(_fulltop)；估计 ~2.2h 墙钟

## Phase 4：真值/评估结果（2026-07-26 23:20-23:45）

### order-2 真值表定稿（990 集合，88 号修正协议）
- synergy2_perconf.csv：11,352 行，**max syn2 = 0.4331**，>0.2 有 220 个，>0.1 有 1168 个
- ⟹ CAISO 二阶强协同丰富（比 PJM 更多），弱的是高阶——"衰减更陡"成为主叙事候选

### ★第一/二幕复刻成功（eval_l1_order2，audit 口径，n=11352）
| 估计器 | ρ | ρ_tail | 备注 |
|---|---|---|---|
| oracle 共享读出 | 0.7466 | 0.5956 | 尾部失明复现 |
| oracle+仿射校准 | 0.7998 | 0.7575 | 2 参数修不满（表达力非尺度）复现 |
| poly2 手工字典 | 0.9035 | 0.7963 | |
| **L0 (last 冻结+闭式)** | **0.9331** | **0.9310** | 零训练超手工字典，复现 |
| cat3 | 0.8890 | 0.9210 | |
| **L1 aug8 s0/s1/s2** | **0.947/0.947/0.945** | **0.969/0.973/0.965** | 阶梯顶端，跨 seed 一致 |
方法阶梯 oracle→affine→poly2→L0→L1 单调，与 PJM 完全同构。

### 闭式共线 R²（协同=共线放大，零成本基线）
- order-2 全空间：ρ=0.728，ρ_tail=0.713，hit20=1.0——**零成本达到 oracle 量级、
  尾部反超 oracle**（0.713 vs 0.596），但距 L0/L1 尚远 ⟹ 线性共线解释大半、
  尾部仍需非线性表征
- L1 o4 top-200 上与 L1 排序 Spearman=0.366（两者在高阶 top 分歧大，待认证裁决）

### 运行中（23:38 启动）
- certs_shard×2（五组认证）+ truth_o3×2（400 三元组）

## Phase 5：认证结果（陆续到达）

### ★o4 L1-top20 + ctrl20 认证（2026-07-27 01:02 完成）
- **精确率 19/20 = 95%**（PJM 单 seed 45%）；ctrl 0/20 >0.10；Mann-Whitney p<0.0001
- L1 几乎无偏且略保守：结构化均值 0.114 → 认证均值 0.128（认证反而更高）
- **o4 最强认证 syn = 0.157**（PJM 0.196）
- 结构：conf=1(发电计划总量) 主导，高频字段 26(bpat 外部负荷)+风光(8/9/13/14/15/17)
  +RT LMP(22/23/24)；conf=3(export) 少数
- 插曲：o3 认证第一版源文件只含 FDR 通过集合→对照池空→shard1 组 3 崩溃
  （组 4/5 被 set -e 连带跳过）。已修：源=通过∪真弱(syn<0.05)，中间地带排除；
  shard1 组 3-5 已补跑（brckjfd9q），shard0 未受影响。
- 环境注：2026-07-27 ~01:30 起 GPU1 被其他用户 19GB 任务共享（REID 训练），
  认证进程（488MB）不受影响；评估类任务改跑 GPU2 或调小 chunk。

### order-3 无偏池 + 评估（2026-07-27 02:30）
- h3_unbiased_pool.csv：400 三元组，随机池 max syn3=0.2446、>0.1 有 42/4800
- E3 阶梯（audit，n=4800）：oracle ρ/tail 0.480/0.424 → poly2 0.802/0.590 →
  full 0.809/0.581 → L0-last 0.820/0.766 → **L1 0.802/0.788**
  （字典全局 ρ 追平 L0/L1，但尾部差 0.17+ ——"字典尾部失明"复刻）
- L1 分档 gap（truth−est）：全部 ≤0.017 —— L1 在 o3 各档几乎无偏
- [ ] order-2 真值表（44 单 + 946 对 = 990 次重训 ≈7.7 GPU·h，最长条目，尽早后台跑）
- [ ] order-3 无偏三元组池（~300 三元组重训）
- [ ] o4 扫描×3seed + o5 扫描（≥seed0）+ full 字典对照扫描
- [ ] paired FDR order-3
- [ ] 认证：L1-top + full-top + 随机对照（o4/o5）
- [ ] 跨 seed 集成过滤 + cross_checks（衰减律/反层级）+ 闭式共线 R² 加验
- [ ] CHANGELOG.md 定稿 + PAPER_STORYLINE 更新第二数据集条目
