# 调查报告：PJM 数据 → 子集条件 oracle 训练链条

## 1. 预处理链条（原始 csv → cleaned csv → 内存中的 44+12）

**关键澄清：预处理是两级的。cleaned.csv 只做了删列；常量列剔除、dropna、切分、标准化都发生在各实验运行时的内存里，不落盘。**

### 1.1 第一级：生成 cleaned.csv（1 号实验）
- 原始文件（两份同内容拷贝）：
  - `/data1/duhaocun/projects/GNN_Aggresvation/pjm_rto_hourly_2025_aligned_processed_one_header.csv`
  - `/data1/duhaocun/projects/GNN_Aggresvation/data/Raw/pjm_rto_hourly_2025_aligned_processed_one_header.csv`（配置实际引用这份）
- 生成代码：`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation1/src/data_processing.py`，函数 `load_and_preprocess_data()`（约 40-63 行）。配置：`DNN_Aggresvation1/configs/config.yaml`（`raw_csv_path: "../data/Raw/..."`）。
- 只做两件事：`df.drop(columns=drop_columns)` + `pd.to_numeric(errors='coerce')`，然后写出 `/data1/duhaocun/projects/GNN_Aggresvation/data/Processed/pjm_rto_hourly_2025_cleaned.csv`（8760 行 × **61 列**，72−11）和签名文件 `pjm_rto_hourly_2025_cleaned.metadata.json`（记录 raw 文件名 + drop 列表，用于缓存校验）。**无归一化、无切分、无删行。**
- 删除的 11 列（metadata.json 原文）：`datetime_beginning_utc, datetime_beginning_ept, net_sched_interchange_mw, prelim_load_avg_hourly, total_pjm_rt_load_mwh, wind_generation_mw, solar_generation_mw, da_as_as_mw_primary_reserve, da_as_as_mw_synchronized_reserve, da_as_as_mw_thirty_minutes_reserve, system_energy_price_rt`

### 1.2 第二级：运行时预处理（69 号，被 73/75/76/91/93/94 全部复用）
- 代码：`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/src/data_processing.py`
  - `load_and_preprocess()`：读 cleaned.csv → `select_dtypes(number)` → **删常量列**（std<1e-10，实际删 5 列：`gen_fuel_storage_pct, da_as_mcp_thirty_minutes_reserve, da_as_ircmwt2_primary_reserve, da_as_ircmwt2_synchronized_reserve, da_as_ircmwt2_thirty_minutes_reserve`）→ **`df.dropna()`：8760 → 6556 行**（删 2204 行）→ 列重排为 [general..., confidential...]。61−5−12 = **44 个 general**。
  - `shuffle_split()`：`np.random.RandomState(seed).permutation`，`train_ratio=0.7` → **train 4589 / test 1967**。
  - `standardize()`：逐列 z-score，**只用训练集的 mean/std**（std=0 替换为 1）。
  - `prepare_data(cfg)`：主入口，返回 train_data/test_data/general/confidential/indices 等字典。
- 各实验 `base.yaml`（75/91 等完全相同）：`csv_path: ../data/Processed/pjm_rto_hourly_2025_cleaned.csv`，`split_mode: shuffle`，`train_ratio: 0.7`，`runtime.seed: 42`。

### 1.3 确切字段列表（我用项目环境 `Pytorch310_codex` 实际复算核对，nG=44, nC=12）

**44 个可见（general）字段，按内存索引 0-43：**
`gen_fuel_coal_mw, gen_fuel_gas_mw, gen_fuel_hydro_mw, gen_fuel_multiple_fuels_mw, gen_fuel_nuclear_mw, gen_fuel_oil_mw, gen_fuel_other_renewables_mw, gen_fuel_solar_mw, gen_fuel_storage_mw, gen_fuel_wind_mw, gen_fuel_coal_pct, gen_fuel_gas_pct, gen_fuel_hydro_pct, gen_fuel_multiple_fuels_pct, gen_fuel_nuclear_pct, gen_fuel_oil_pct, gen_fuel_other_renewables_pct, gen_fuel_solar_pct, gen_fuel_wind_pct, forecast_load_mw_latest_available, forecast_load_mw_day_ahead, rmccp, rmpcp, total_pjm_loc_credit, total_pjm_reg_purchases, total_pjm_self_sched_reg, total_pjm_assigned_reg, total_pjm_rmccp_cr, total_pjm_rmpcp_cr, da_as_mcp_primary_reserve, da_as_mcp_synchronized_reserve, da_as_as_req_mw_primary_reserve, da_as_as_req_mw_synchronized_reserve, da_as_as_req_mw_thirty_minutes_reserve, da_as_ss_mw_primary_reserve, da_as_ss_mw_synchronized_reserve, da_as_ss_mw_thirty_minutes_reserve, da_as_nsr_mw_primary_reserve, system_energy_price_da, total_lmp_rt, marginal_loss_price_rt, net_inadv_interchange_mw, gross_sched_interchange_mw, gross_inadv_interchange_mw`

**12 个机密（confidential）字段（base.yaml `fields.confidential`，内存索引 44-55）：**
`net_actual_interchange_mw, gross_actual_interchange_mw, total_gen, metered_load_mw, total_losses, congestion_price_da, congestion_price_rt, marginal_loss_price_da, total_lmp_da, da_as_total_mw_primary_reserve, da_as_total_mw_synchronized_reserve, da_as_total_mw_thirty_minutes_reserve`

## 2. Oracle 训练

### 2.1 架构（69 号定义，全线复用）
- `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/src/oracle.py`，类 `MLPOracle(n_general, n_confidential, hidden=256, dropout=0.1)`：
  - 输入 `torch.cat([x*m, m], dim=1)`（**2×44=88 维**），3 个隐层 `Linear(·,256)+ReLU+Dropout(0.1)`，输出 `Linear(256,12)`。Sequential 下标：ReLU 在 1/4/7（91 号 `RELU_IDX` 依赖此约定）。
  - 同文件还有 `GNNOracle`（相关图先验+可见性感知注意力），但 **74 号已判定图结构放弃**，75 号起只训 MLP。

### 2.2 两段式子集采样（73 号设计 = 69 号 `sample_mask`）
- 正典实现：`oracle.py:sample_mask(batch, n_general, rng)`（122-129 行）：**先抽保留数 k ~ U{1..44}，再从 44 个可见字段中不放回均匀抽 k 个置 1**。73 号 `train_variants.py` 里叫 `ours`，75 号 `src/samplers.py` 里叫 `s_uniform`（注释明确"必须与 69/src/oracle.py:sample_mask 逐位相同"）。
- 75 号扫了 9 种尺寸分布（`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/src/samplers.py`，`SAMPLERS` 字典）：`none`(恒满)、`bern50`(逐字段 Bernoulli(0.5)，空掩码修补)、`uniform`(两段式)、`logunif`(k≈exp U(0,ln nG))、`small50/small80`(50%/80% 概率 k~U{1..4} 否则 U{1..nG})、`workload`(50% U{1..3} + 50% U{13..44})、`workload_hard`(90/10)、`large50`(50% U{14..44})。

### 2.3 训练脚本与超参（75 号）
- `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/scripts/train_variant.py`：
  - `EPOCHS=400, PATIENCE=60, BATCH=256, LR=1e-3, WD=5e-4`（Adam），MSE 损失；数据切分 seed 42 固定；**训练集内部再切 15% 做验证**（`np.random.RandomState(args.seed).permutation`）；早停用 **8 组固定验证掩码**（`N_VAL_MASK=8`，`val_rng=RandomState(999)`；`none` 方案只用 1 组）；训练掩码 rng = `RandomState(1234+seed)`；模型初始化 seed = `args.seed`。
  - **训练很快**：logs 显示单个 oracle 11-36 秒（GPU），如 `[none seed0] ep=352/400 [11s]`。
- Checkpoint：`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/outputs/oracle_{scheme}_seed{0,1,2}.pt`，共 9×3=27 个。保存字典：`{state, scheme, seed, nG, nC, val_loss, r2_full, epochs_run, n_params}`。
- 73 号的消融版 checkpoint 在 `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation73/outputs/oracle_{mlp,gnn}_{none,bern50,ours}_seed0.pt`（脚本 `DNN_Aggresvation73/scripts/train_variants.py`，同一套超参）。

### 2.4 91 号 L0 用的"现成 oracle"
- **`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt`**。硬编码常量 `CKPT` 出现在 91 号 4 个脚本：`order2_truth.py:41`、`order3_truth.py:38`、`readout_vs_feature.py:45`、`inject_l0.py:55`、`bench_cost.py:37`（`order2_truth.py` 支持 `--ckpt` 换 seed0/1/2 做稳健性，输出 csv 带 `ckpt` 列）。
- 加载函数：`DNN_Aggresvation91/src/featridge.py:load_oracle()`；特征提取类 `FrozenPhi`（`last`=最后一层 ReLU 256 维；`cat3`=三层拼接 768 维）；闭式岭读出 `ridge_r2()`，alpha 网格 `ALPHAS=(1e-3,1e-2,0.1,1,10,100)`，逐 set×conf 独立选 alpha。
- 93 号 L1（闭式头进训练回路）：`DNN_Aggresvation93/scripts/train_l1.py`，主干与 MLPOracle 同构同参数量，`EPOCHS=400, PATIENCE=60, LR=1e-3, WD=5e-4`，每 epoch 8 个 meta-step，每步 4 掩码 × (512 support + 256 query)，λ 可学（log 参数化），`--aug N` 加合成目标；产物 `DNN_Aggresvation93/outputs/oracle_l1_aug8_seed{0,1,2}.pt`、`oracle_l1_seed0.pt`、`oracle_l1_long_seed0.pt`。

## 3. 75 号 1576 子集真值表

- 生成脚本：`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/scripts/build_evalset.py`。汇总三个来源并按 `frozenset(fields)` 去重（1437+85+60−6 重复=1576）：
  - 69 号 `outputs/truth_long.csv`：1437 = 44 单字段 + 946 对 + 397 三元组 + 50 wide_random；
  - 68 号 `retrain/bt*.json`：85 个中段（|S| 5-25）；
  - 75 号新补 `retrain/lg*.json`：60 个大集合（|S| 17-43，由 `gen_truth_subsets.py` 生成，`MASTER_SEED=7501`，四带 [17-24]/[25-32]/[33-38]/[39-43] 各 15 个，重训 worker=`retrain_worker.py`：DNN hidden=128 两隐层 dropout 0.15，EPOCHS=400 PATIENCE=120，Adam 1e-3/5e-4，batch 128；注意其 `train_generic` **用测试集早停**——脚本注释自认是已知问题，为与既有 1437 真值口径一致而保留）。
- 输出文件与格式：
  - `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/outputs/evalset.json`：`{sid: {group, size, fields}}`，1576 键（sid 如 `p00_01`）；
  - `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75/outputs/truth_all.csv`：18912 行（=1576×12），列 `sid, group, size, band, conf, truth`（band ∈ 1-2/3-4/5-8/9-16/17-32/33-44）；`truth` 为该子集重训 DNN 后单个 conf 的测试 R²。
- 后续使用：75 号自己的 `eval_variant.py/analyze.py`（est_*.csv 与 truth 对齐算 MAE/Spearman），76 号三个脚本（`fine_ksweep.py`、`analyze_f1.py`、`simulate_f2.py`）直接读 `../DNN_Aggresvation75/outputs/{evalset.json, truth_all.csv}` 做微调相变分析。77 号之后的高阶实验转用各自的专用重训真值（如 68 号 `synergy2_perconf.csv`）。

## 4. 切分协议（三层）

1. **数据集切分**：shuffle（非时间切分），`train_ratio=0.7`，seed=42（`runtime.seed`），6556 行 → 4589 train / 1967 test；标准化只用 train 统计量。全部实验共用这一份切分。
2. **oracle 训练内部**：train 的 15% 做验证/早停（seed=训练 seed，0/1/2）。93 号 L1 同协议。
3. **search/audit 分片**（88 号引入，91/93/94 沿用）：`featridge.py` 中 `SPLIT_SEED=880725` 对 1967 行 test 做 permutation，**前一半(983)=search、后一半(984)=audit**（`order2_truth.py:99` `a_idx=perm[len//2:]`）；闭式读出的 alpha 选择用 train 内 fit/val 分片，`VAL_SEED=20260724`（与 85/90 号一致），15% val。评估集从不参与 alpha 选择或拟合。

## 5. 重训真值（认证）历史文件

- **91 号** `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation91/outputs/`：
  - `order2_truth.csv`：列 `kind,n,rho,rho_tail,hit20,hit100,med_rank_true_top10,n_strong,sec,ckpt`（kind∈oracle/oracle_affine/poly2/last/cat3/…；真值源=68 号 `outputs/synergy2_perconf.csv`）；
  - `order3_truth.csv`：同类 + gap 分档列；`readout_vs_feature.csv`：逐 `(i,j,conf)` 行，列 `i,j,conf,band,method,v_pair,syn,truth`；`inject_l0_*.csv`、`bench_cost_o3.csv`。
- **93 号** `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation93/outputs/`（认证脚本 `scripts/certify_highorder.py`：每个五元组重训 1+5=6 个专用 DNN，`syn5_true=max_c[v(S)_c−max_{|T|=4}v(T)_c]`，EPOCHS=400/PATIENCE=120，train 内 val 早停即 88 号修正口径，N_TOP=40/N_CTRL=20，PICK_SEED=930726，断点续跑逐行追加）：
  - 逐集合明细：`certify_o5_s{0..3}of4.csv`、`certify_o5_closed_s*.csv`、`certify_o5_l1top_s*.csv`、`certify_o4_l1top_s*.csv`、`certify_o4_fulltop_s*.csv`；列：`S,group,artifact,syn_struct,conf_struct,v_struct,syn_true_audit,conf_true_audit,v_true_audit,maxsub_true_audit,syn_true_full,conf_true_full,v_true_full,syn_at_struct_conf,retained,sub_exceeds`（audit 分片与全 test 两个口径并列）；
  - 合并/汇总：`certify_merged.csv`、`certify_l1top_merged.csv`、`certify_summary.csv`（中文列：`组,n,结构化syn,认证syn,认证syn最大,保留率,过0_2的,过0_1的,子集反超`——即"90 号强五阶保留率 0.083、与 control 无差异"的原始记录）；配对 FDR：`paired_fdr_o3_cat3.csv`（分析脚本 `analyze_certify.py`、`paired_fdr.py`）。
- **94 号** `/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation94/outputs/`：
  - `decay_law_certified.csv`：真值版衰减律定稿表（列 `阶,最强真值,口径,强协同数`：3 阶 0.455 / 4 阶 0.196 / 5 阶 0.154）；
  - `ensemble_filter_o4.csv`：跨 seed 秩一致性过滤明细（在 93 号列基础上加 `Stup,rank_s1,rank_s2,cross_rank,true`）；`hierarchy_check_o5.csv`：反层级认证级测量。

**补充**：91 号所有闭式对照的二阶外部真值统一是 `DNN_Aggresvation68/outputs/synergy2_perconf.csv`；93 号 `eval_l1_order2.py` 同源。运行环境为 conda env `Pytorch310_codex`（`/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python`）。