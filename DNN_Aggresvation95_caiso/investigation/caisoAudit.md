# CAISO 2025 小时数据体检报告（第二验证数据集适配性评估）

数据：`/data1/duhaocun/projects/GNN_Aggresvation/data/caiso/caiso_extracted/caiso_2025_hourly_augmented/caiso_2025_hourly_aligned_augmented.csv.gz`
实测 8760 行 × 307 列（2 时间列 + 305 数值列），与 `validation_report.md` 声明一致。

---

## 1. 列的语义分组（305 数值列，共 11 组）

| 组 | 列数 | 维度结构 | 含义 |
|---|---|---|---|
| `actual_load` | 35 | 按 TAC_AREA（34 BA + `ca_iso_tac` 总量） | 实际小时负荷（SLD_FCST/ACTUAL）。含 CAISO 内部 5 个 TAC（pge_tac、sce_tac、sdge_tac、vea_tac、mwd_tac）+ 29 个 WEIM 外部 BA（bpat、pace、nevp、azps、srp、ladwp、banc 系列、walc 系列等） |
| `dam_load_forecast` | 34 | 同上但无 `bcha` | 日前市场负荷预测（SLD_FCST/DAM） |
| `actual_renewable_generation` | 6 | 3 hub（np15/sp15/zp26）× {solar, wind} | 实际风光发电（SLD_REN_FCST/ACTUAL） |
| `dam_renewable_forecast` | 6 | 同上 | 日前风光预测 |
| `dam_lmp` | 15 | 3 hub × 5 成分（lmp/energy/congestion/loss/ghg） | 日前 LMP 及官方分量（PRC_LMP v12） |
| `rt15_lmp_hourly_mean` | 15 | 同上 | RTPD 15 分钟 LMP 的小时算术均值（PRC_RTPD_LMP，四个 quarter 全齐才取均值，覆盖 8760/8760） |
| `dam_schedule` | 20 | 4 流向（generation/load/import/export）× 区域（caiso_totals + tac_north/ecntr/south/ncntr + nontac） | ENE_SLRS：日前计划的发电、负荷、进出口——**这是 CAISO 版的"总发电量/interchange"** |
| `dam_as_requirement` | 52 | 6 AS 区域（caiso/np26/sp26 及各自 _exp）× 4 产品 × {min,max} = 48，+ caiso_exp 的 mileage up/down × {min,max} = 4 | 日前辅助服务需求上下限（AS_REQ） |
| `dam_as_clearing_price` | 26 | 4 产品（ru/rd/sr/nr）× 6 区域 + rmu/rmd（仅 caiso_exp） | AS 清算价（PRC_AS） |
| `dam_as_total_procured` | 26 | 同上 | AS 总采购 MW（AS_RESULTS，仅 AS_MW） |
| `rtpd_eim_transfer` | 70 | {import, export} × 15 个 EIM BA × tie（共 35 个 tie 对） | WEIM 实时转移（ENE_EIM_TRANSFER_TIE，仅 CISO 侧） |

合计 35+34+6+6+15+15+20+52+26+26+70 = 305 ✓

---

## 2. 数据质量

**时间轴完整**：UTC 2025-01-01 00:00 → 2025-12-31 23:00，8760 行，相邻间隔恒为 1 小时（8759 个差分全部 = 1h），0 重复、0 缺口。

**缺失**（总缺失格 10,511 / 2,672,400 = 0.39%）：
- 缺失 >5% 的列**只有 1 列**：`dam_schedule__mw__import__tac_ncntr` 缺 98.90%（8664 行）——官方文件本身只发布了 96 小时，必须剔除。
- 70 列 EIM 各缺 25 小时（0.29%，来自 quarter-hour 不齐）；`dam_load_forecast__mw__pace` 缺 24 行（0.27%）；6 列 renewables 各缺 12–13 行（0.14%）。其余 227 列零缺失。

**全零列：47 列**（全部应剔除）：
- `actual_load__mw__avrn`、`dam_load_forecast__mw__avrn`（2）
- 6 列 ghg 成分（DAM+RT × 3 hub，2025 年全为 0）
- 18 列 AS requirement `maximum`（non_spinning/regulation_up/spinning × 6 区域；注意 regulation_down 的 maximum 非零）
- 7 列 AS clearing price（nr 三区、sr 三区、rd np26）
- 14 列 EIM tie（如 banc lake/standiford/tesla230、bcha import 侧、ava、srp marketplace）

**近零列（≥99% 为 0）另有 10 列**（7 列 AS 清算价 + 3 列 EIM tie），信息量极低。

**去冗余后的有效独立维度约 233 列**（305 − 47 全零 − 1 超高缺失 − 约 24 列被下述精确恒等式/完全重复决定）。

其他质量观察：负荷预测质量总体很好（外部 BA actual-vs-DAM 相关 0.97–0.99），但 `sce_tac` 仅 0.768、`ca_iso_tac` 仅 0.859，偏低但真实。价格含丰富鸭子曲线信号：DAM sp15 负价 976 小时、RT sp15 负价 955 小时，RT np15 max $413.37。注意 `pge`（Portland General Electric，WEIM BA）与 `pge_tac`（PG&E）是不同实体（相关仅 0.525），命名易混。

---

## 3. 精确线性恒等式（全部实测验证）

这是本数据集最重要的结构特征，**切分可见/机密字段时必须逐条审计**：

1. **LMP 恒等式精确成立**：`lmp = energy + congestion + loss + ghg`，6 组（DAM/RT × 3 hub）max|残差| = 1.0e-5（= 官方发布的 5 位小数舍入精度），mean|残差| ≈ 2–3e-6。与官方附带的 `lmp_identity_validation.csv` 完全一致。→ 6 条总价列是分量的精确线性组合。
2. **energy 分量跨 3 个 hub 完全相同**（max|diff| = 0.0000，DAM 和 RT 各如此）→ 每个市场实际只有 1 条 energy 序列，4 列是纯副本。
3. **负荷加和恒等**：`actual_load ca_iso_tac = pge_tac + sce_tac + sdge_tac + vea_tac + mwd_tac`，残差 **0.0000**；DAM 版 max|残差| = 0.02（舍入）。外部 BA 负荷与该恒等式无关。
4. **dam_schedule 加和恒等**：4 类流向的 `caiso_totals` = 各 TAC 子区之和，残差全为 0（load 含 nontac + ncntr，generation/import 含 ncntr）。
5. **AS 总采购加和恒等**：`as_caiso = as_np26 + as_sp26`（ru/rd/sr/nr 四个产品，max|残差| 0.01）；`_exp` 版本同样成立。另有完全重复列：nr 的三个区域 base == exp 逐值相同、rd 的 np26 == np26_exp。
6. **不成立的**（可放心当独立信息用）：AS requirement `minimum` 不满足区域加和（caiso − (np26+sp26) max 达 581 MW）；`gen + import − export − load` 能量平衡不成立（mean|imbalance| ≈ 1075 MW，ENE_SLRS 各项口径不同）；EIM 净转移与 DAM 计划净进口相关仅 0.44（两者是不同市场层的量）。

**近似恒等（无法消除，需在实验注记）**：hub 间 LMP 相关 sp15–zp26 = 0.982（DAM）/0.959（RT），np15–sp15 = 0.872/0.761；若"分区可见 + 总量机密"，因 vea+mwd 仅占总负荷 0.83%，3 个大 TAC 可见即可把机密 `ca_iso_tac` 重构到 R²>0.99——这与 PJM（zone 加总 ≈ RTO 负荷）性质相同，不是 CAISO 独有缺陷，但要写进实验协议。

---

## 4. 与 PJM 语义对齐度

PJM 12 个机密字段的 CAISO 对应：

| PJM 机密字段 | CAISO 对应列 | 对齐质量 |
|---|---|---|
| metered_load | `actual_load__mw__ca_iso_tac` | ✓ 完全对应 |
| total_gen | `dam_schedule__mw__generation__caiso_totals` | △ 只有**日前计划值**，无 RT 实测总发电 |
| gross/net interchange | `dam_schedule__mw__import__caiso_totals` / `__export__caiso_totals`（差 = net） | △ 日前计划口径；RT 只有 EIM 转移（70 列 tie 级，可求和成净转移，但只覆盖 WEIM 部分） |
| losses (MW) | **无对应列** | ✗ 缺失（只有 loss 价格分量） |
| total_lmp_da | `dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd`（或 np15/zp26） | ✓ hub 级 |
| DA congestion price | `dam_lmp__congestion_usd_per_mwh__th_*` | ✓ |
| RT congestion price | `rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_*` | ✓ |
| marginal loss price | `dam_lmp__loss_usd_per_mwh__th_*`（RT 版也有） | ✓ |
| AS total MW × 3 | `dam_as_total_procured__mw__{ru,rd,sr,nr}__as_caiso` | ✓ 且有 4 个产品可选 3–4 |

可见侧对应：load actual/forecast ✓（且比 PJM 多出 29 个外部 BA 的空间维度）、renewables actual/forecast ✓（hub×solar/wind 6+6）、LMP DA/RT 全分量 ✓、AS requirement/clearing price ✓（PJM 未必有这么全）。

**CAISO 缺什么**：① RT 实测总发电量；② 全口径实测 interchange（EIM 转移只覆盖 WEIM）；③ MW 损耗；④ RT 负荷预测（本表只提取了 ACTUAL 和 DAM 两个 market_run）。**CAISO 多什么**：EIM tie 级实时转移、AS 需求上下限、dam_schedule 的 TAC 分区计划、ghg 分量（虽全零）。

---

## 5. 字段收缩方案（三个候选）

先统一执行**剔除底线**：47 全零列 + `dam_schedule__mw__import__tac_ncntr` + 4 条 energy 副本（每市场留 1 条）。

### 方案 A「系统级镜像 PJM」（推荐，44 可见 + 12 机密）

**机密 12 列**（与 PJM 逐项对位）：
```
actual_load__mw__ca_iso_tac
dam_schedule__mw__generation__caiso_totals
dam_schedule__mw__import__caiso_totals
dam_schedule__mw__export__caiso_totals
dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd
dam_lmp__congestion_usd_per_mwh__th_sp15_gen_apnd
dam_lmp__loss_usd_per_mwh__th_sp15_gen_apnd
rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd
dam_as_total_procured__mw__ru__as_caiso
dam_as_total_procured__mw__rd__as_caiso
dam_as_total_procured__mw__sr__as_caiso
dam_as_total_procured__mw__nr__as_caiso
```
**可见 44 列**：
- 负荷 7：`actual_load__mw__{pge_tac, sce_tac, sdge_tac}`；`dam_load_forecast__mw__{pge_tac, sce_tac, sdge_tac, ca_iso_tac}`
- 风光 12：6 条 `actual_renewable_generation` + 6 条 `dam_renewable_forecast`（全保留）
- 价格 8：`dam_lmp__energy__th_sp15`（每市场唯一 energy）、`rt15__energy__th_sp15`、`dam_lmp__lmp__{np15, zp26}`、`rt15__lmp__{np15, sp15, zp26}`、`rt15__loss__th_np15`
- 计划负荷分区 3：`dam_schedule__mw__load__{tac_north, tac_ecntr, tac_south}`
- AS 8：`dam_as_clearing_price__usd_per_mw__{ru,rd,sr,nr}__as_caiso` + `dam_as_requirement__mw__as_caiso__{regulation_up, regulation_down, spinning_reserve, non_spinning_reserve}__minimum`
- EIM 大 tie 6：`rtpd_eim_transfer__mw__export__bcha__malin500`、`export__banc__ranchoseco`、`import__nevp__eldorado230`、`export__srp__pvwest`、`import__ladwp__sylmar`、`export__pge__malin500`

已做泄露审计：机密 `dam_lmp sp15` = 可见 energy + 机密 cong + 机密 loss——恒等式两个未知数落在机密侧，从可见集**不可精确重构**任何机密列；可见集内部无完整恒等式四元组；`dam_schedule` 子区只放 load（generation/import/export 子区不可见，避免"和恒等式"泄露机密 totals）。

### 方案 B「WEIM 空间图版」（51 可见 + 12 机密，发挥 GNN 空间结构）

机密同 A。可见 51：外部 12 大 BA actual load（bpat, pace, nevp, azps, srp, psei, ladwp, pacw, ipco, banc, pnm, nwmt）+ 其中 6 个的 DAM forecast（bpat, pace, nevp, azps, srp, psei）+ 内部 3 TAC actual + 3 DAM + 6 条 actual renewables + 价格 5（两条 energy + 3 条 RT lmp）+ EIM 活跃 tie 12 条（上面 top-20 表中取前 12 条非零对）+ AS 清算价 caiso 4 条。特点：节点多（15+ BA），最贴合图聚合实验；代价：外部 BA 与 CAISO 机密量的耦合较弱、偏离 PJM 的 44+12 设定。

### 方案 C「价格全息版」（43 可见 + 11 机密）

机密 = 方案 A 的前 11 条（去掉 nr）。可见 43：12 风光 + 7 负荷（同 A）+ `dam_schedule load` 3 子区 + `dam_schedule generation {tac_north, tac_ecntr}` 2 条（**不放** south/ncntr，否则 generation totals 被和恒等式近似重构到 99.8%）+ 价格 11（dam lmp/cong/loss 的 np15、zp26 各 6 条 + dam energy + rt lmp 3 hub + rt energy）+ AS 8（同 A）。特点：可见侧带完整 DA 分量结构，考验模型对价格分解的利用；风险：np15/zp26 全分量可见后，sp15 机密分量的近似可重构性最高（hub 相关 0.98）。

---

## 6. 最终判定：**调整后适合**

**支持面**：8760 行严格完整连续；核心组（load、renewables、LMP、AS 价格/采购、schedule totals）全年零缺失；语义组与 PJM 的 44+12 设定能一一对位（12 个机密字段全部找得到载体）；额外的 WEIM 空间维度是 PJM 没有的加分项；元数据（逐列字典、SHA-256 溯源、恒等式验证文件）质量很高。

**硬伤（3 条，均可绕开但必须写进协议）**：
1. **"运行敏感量"降级为日前计划口径**：total_gen 和 interchange 只有 DAM 计划值（ENE_SLRS），没有 RT 实测总发电和全口径实测联络线潮流；MW 损耗完全缺失。机密字段的"敏感性"语义比 PJM 弱一档——这是最实质的差异，若审稿人追问需承认。
2. **精确恒等式密布**（LMP 恒等 6 组、负荷/计划/AS 三类加和恒等、energy 跨 hub 复制），305 列名义维度实际只有约 233 个独立维度；任意切分若不做泄露审计，机密字段可能被可见字段**零误差重构**（上面三个方案已逐条规避精确重构；近似重构 R²>0.99 的和恒等式与 hub 相关性无法消除，与 PJM 性质相同）。
3. **57 列垃圾列**（47 全零 + 10 近零）和 1 列 98.9% 缺失必须先剔除，否则常量列会污染标准化和敏感度估计。

推荐落地：以**方案 A** 为主（与 PJM 可比性最强），方案 B 作为图结构消融的补充配置。

分析脚本：`/data1/duhaocun/.cache/vscode-tmp/claude-1019/-data1-duhaocun-projects-GNN-Aggresvation/c13972b3-b4be-4aff-97ec-154b0816a868/scratchpad/caiso_check{,2,3,4}.py`（用 `/data1/duhaocun/miniconda3/envs/Pypower/bin/python` 运行，pandas 2.3.3）。