已完成对两个实验目录及其上游依赖（69/75/68/77/90 号）的完整代码审读。以下是技术报告。

---

# DNN_Aggresvation93/94 管线技术报告（供 CAISO 复刻用）

## 0. 总体架构与目录关系

- **93 号**（`/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation93/`）：核心实现号，含全部可执行脚本与原始输出（L1 训练、全枚举扫描、认证、FDR）。
- **94 号**（`.../DNN_Aggresvation94/`）：纯分析收束号，**零 GPU**，全部读 93 号 `outputs/` 落盘数据。`base.yaml` 与 93 号逐字节相同（已 diff 验证），`src/` 是 93 号 `src/` 的拷贝。
- 两号的 `src/` 只有三个文件：`featridge.py`（闭式岭读出核心）、`fstat.py`（偏回归 t 检验/BH，93 号实际主线未用，属 90 号遗留工具）、`runlog.py`（写 `RUNLOG.md` + `outputs/events.jsonl` 的日志器，注意其 ROOT 是相对自身路径解析的，复制目录即可用）。
- **关键外部依赖（跨目录 import/读文件）**：
  - `DNN_Aggresvation69/src/data_processing.py::prepare_data` 与 `DNN_Aggresvation69/src/oracle.py::MLPOracle, sample_mask` —— 所有脚本都通过 `sys.path.insert(0, str(REPO/"DNN_Aggresvation69"))` 导入（`REPO = 脚本目录的 parents[2]`，即项目根）；
  - `DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt` —— L0 基线 checkpoint（评估/对照用，L1 训练不依赖它）；
  - `DNN_Aggresvation68/outputs/synergy2_perconf.csv` —— 二阶重训真值（11352 条 (i,j,conf)），仅评估用；
  - `DNN_Aggresvation77/outputs/h2_unbiased_pool.csv` —— 三阶无偏重训真值池（`group=="triple_rand"`，列 `ix/conf/syn3_true`），仅评估用；
  - `DNN_Aggresvation90/outputs/rescan_o{3,4,5}_full_audit_s*.parquet` —— full 手工字典的扫描结果，仅作对照组（`certify_highorder.py` 不传 `--source` 时默认读它）。
  - **CAISO 复刻的最小闭环不需要 68/77/90 的文件**（它们是历史对照/评估参照）；必须有的只有 69 号的 `src/` 和一份新数据 CSV。

---

## 1. 完整可执行管线（按执行顺序）

所有入口都在 `DNN_Aggresvation93/scripts/`（94 号的 4 个分析脚本零 GPU 后置）。共同点：每个脚本开头读 `ROOT/base.yaml` 后**立刻用硬编码路径覆盖** `cfg["dataset"]["csv_path"]`（见 §5），然后 `prepare_data(cfg)` 得到标准化后的 train/test 数组。

### 第 1 步：训练 L1 摊销模型 —— `scripts/train_l1.py`
- 命令行：`--seed`(默认0) `--aug`(默认0，主线用 8) `--tag`(输出文件后缀) `--epochs`(默认400，实际主线传 2500 靠早停) `--steps`(默认8)
- 实际主线命令等价于：`python scripts/train_l1.py --seed 0 --aug 8 --tag _aug8 --epochs 2500`（seed 1/2 同理）
- 输入：仅数据 CSV（经 prepare_data）。不需要任何 checkpoint。
- 输出：`outputs/oracle_l1{tag}_seed{seed}.pt`（dict：`state/seed/nG/nC/val_loss/epochs_run/aug/lam`）
- 实测时长（RUNLOG.md）：seed0 aug8 早停 239ep/66s；seed1 832ep/234s；seed2 162ep/45s；单卡。

### 第 2 步（可选，评估）：`scripts/eval_l1_order2.py` / `eval_l1_order3.py`
- 参数：`--kinds`（估计器列表：oracle/oracle_affine/poly2/full/last/cat3/last+p2/cat3+p2）`--chunk 64` `--ckpt`（传第 1 步的 .pt）`--tag`
- 输入：ckpt + 68 号二阶真值 / 77 号三阶无偏池（**PJM 专属，CAISO 上需自建或跳过**）
- 输出：`outputs/eval_l1_order2{tag}.csv`（列 kind/n/rho/rho_tail/hit20/hit100/med_rank_true_top10/n_strong/sec/ckpt）、`eval_l1_pairs{tag}.csv`；order3 同理。
- 时长：2–23s/kind。

### 第 3 步：全枚举扫描 —— `scripts/scan_order5.py`（名字叫 order5，实际 `--order` 通用，o4 也是它跑的）
- 参数：`--ckpt`(必填) `--kind`(默认cat3，主线用 **last**) `--order`(默认5) `--shard/--nshard`(分片并行) `--chunk 64` `--tag` `--subs_only`(只算 m−1 阶子集缓存)
- 主线流程（RUNLOG 可复原）：
  1. `--subs_only --nshard 1` 先算 C(44,4)=135,751 个四阶子集 v 值 → 缓存 `outputs/subs_o4_{ckpt.stem}_{kind}{tag}.npy`（294s 单卡）；
  2. 4 个进程各 `--shard i --nshard 4`（各挂一张卡）枚举 C(44,5)=1,086,008 五元组，读缓存 → `outputs/scan_o5_{ckpt.stem}_{kind}_s{i}of4.parquet`（列 `indices`(字符串元组)/`syn`/`conf`/`v_set`），583–1050s/分片。
  3. o4 同理：先 `subs_o3` 缓存（13,244 个，29s），再 2 分片扫 135,751 四元组（280s/分片）。
- syn 定义：`syn(S) = max_c [ v(S)_c − max_{T⊂S,|T|=m−1} v(T)_c ]`，v 为闭式岭读出在 audit 分片上的 R²（`clamp_min(0)`）。

### 第 4 步：配对 FDR —— `scripts/paired_fdr.py`
- 参数：`--order 3` `--kind cat3` `--chunk 48` `--q 0.05` `--taus 0,0.05,0.10,0.15` `--max_sets`(调试)
- 输入：**模块级常量 `CKPT = DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt`**（注意：FDR 用的是 L0 冻结 oracle，不是 L1 ckpt，且无 `--ckpt` 参数——CAISO 上要么改这个常量指向新 L1/oracle，要么加参数）。
- 输出：`outputs/paired_fdr_o{m}_{kind}.csv`（列 `S/conf/syn_search/syn_audit/se/p@τ/q@τ/rej@τ ×4`）
- 时长：order-3 全扫 13,244 集合 + 946 子集，497–629s 单卡。
- 协议：test 集经 `SPLIT_SEED=880725` 置换一分为二：**search 半（983 行）只做选择**（每个 S 选哪个 conf、哪个最强父集 T），**audit 半（984 行）只算一次 p 值**。统计量 `d_i = err²_T(i) − err²_S(i)`（同一样本配对），`syn = mean(d)/Var(Y_c)`，`t = (syn−τ)/se`，单边 p（df = n_audit−1），BH-FDR q=0.05。逐 τ 检验 `H0: syn ≤ τ`（τ=0 是无筛选力的点零假设，实际用 τ=0.10）。

### 第 5 步：DNN 重训认证 —— `scripts/certify_highorder.py`
- 参数：`--order 5` `--shard/--nshard` `--seed 0`(重训随机性) `--source`(扫描产出前缀，如 `scan_o5_oracle_l1_aug8_seed0_last`；**不传则读 90 号 full 字典 rescan**) `--n_top 40` `--n_ctrl 20` `--tag`
- 主线三组认证命令等价：
  - full-top：`--order 5 --n_top 40 --n_ctrl 20`（无 source，读 90 号）→ `certify_o5_s{i}of{n}.csv`
  - L1-top：`--order 5 --source scan_o5_oracle_l1_aug8_seed0_last --n_top 21 --n_ctrl 0 --tag _l1top` → `certify_o5_l1top_s*.csv`
  - o4 两组：`--order 4 --source scan_o4_... --tag _l1top` / 无 source `--tag _fulltop`
- 输出 CSV 列：`S,group,artifact,syn_struct,conf_struct,v_struct,syn_true_audit,conf_true_audit,v_true_audit,maxsub_true_audit,syn_true_full,conf_true_full,v_true_full,syn_at_struct_conf`
- 断点续跑：逐集合 append 写 CSV；启动时 glob `certify_o{order}_s*of*.csv` 汇总已完成的 S 集合并跳过（注意此 glob 不含 tag，即带 tag 的文件不进 done 集，但 untagged 的 60 个会被 tagged 运行识别跳过——RUNLOG 里"已完成 60"即此机制）。
- 后处理：`scripts/analyze_certify.py`（无参数，聚合 `certify_o5_s*of*.csv` → `certify_merged.csv`/`certify_summary.csv`，Mann-Whitney U + Spearman + 子集反超诊断）；`scripts/l0_on_certified.py`（在已认证集合上对比各估计器）。

### 第 6 步（94 号，零 GPU 分析）：
- `DNN_Aggresvation94/scripts/cross_checks.py`：A) L1 对 full 假 top 的排名否决；B) 反层级/beam 漏检率 → `outputs/hierarchy_check_o5.csv`；C) 真值版衰减律 → `outputs/decay_law_certified.csv`。无参数，硬编码读 93 号 parquet/csv。
- `DNN_Aggresvation94/scripts/seed_agreement.py`：多 seed 稳健性（见 §4）。
- `DNN_Aggresvation94/scripts/ensemble_filter.py`：跨 seed 集成过滤 → `outputs/ensemble_filter_o4.csv`。
- `make_figs.py`（两号各一）：出图。

### CAISO 最小复刻序：
1) 建 CAISO 版 base.yaml + 改各脚本 csv_path（§5）→ 2) `train_l1.py` ×3 seed → 3) `scan_order5.py` 对 seed0 全枚举 o4/o5（seed1/2 至少扫 o4 供过滤用）→ 4) `paired_fdr.py --order 3`（CKPT 改为新模型）→ 5) `certify_highorder.py` 认证 L1-top-K + 随机对照 → 6) `ensemble_filter.py`/`cross_checks.py` 改路径后运行。

---

## 2. L1 模型架构细节（`scripts/train_l1.py` + `src/featridge.py` + 69 号 `src/oracle.py`）

**主干（与 75 号 MLPOracle 完全同构，保证可比）**：
```
MLPOracle(n_general, n_confidential, hidden=256, dropout=0.1)
net = Sequential(
  Linear(2*nG, 256), ReLU, Dropout(0.1),   # ReLU 下标 1
  Linear(256, 256),  ReLU, Dropout(0.1),   # ReLU 下标 4
  Linear(256, 256),  ReLU, Dropout(0.1),   # ReLU 下标 7  ← φ ("last", 256 维)
  Linear(256, nC))                          # 共享读出，L1 训练/查询均不经过
```
- **输入**：`concat([x⊙m, m])`，2·nG 维（PJM 上 88 维）。x 是标准化后的 general 字段行向量，m 是 0/1 可见性掩码。φ 的两种取法在 `featridge.FrozenPhi`：`last`=net[7] 之后的激活（256 维，L1 训练目标对应此配置）；`cat3`=三层 ReLU 激活拼接（768 维，eNTK 廉价代理）。`RELU_IDX=(1,4,7)`、`train_l1.RELU_LAST=7` 均与该结构绑定。
- **闭式头**（`train_l1.closed_form_loss`，训练回路内可微）：support 特征按批内均值/方差标准化后，`β = solve(PᵀP + λI, Pᵀ(y−ȳ))`，query 损失 `MSE(y_q − ȳ − Qβ)`；**梯度反传穿过 `torch.linalg.solve`**；λ 以 `log_lam` 参数化（初始 e⁰=1.0）与主干一起进 Adam。
- **训练循环**：每 epoch 8 个 meta-step（`N_STEP=8`）；每 step 采 `B_MASK=4` 个掩码，每掩码从 train 内部训练份里无放回抽 `N_SUP=512` support + `N_QRY=256` query 行。**掩码采样分布 = 69 号 `sample_mask`：先抽保留数 k~U{1..nG}，再均匀无放回抽 k 个可见字段**（各子集尺寸带均衡覆盖，就是记忆里"两段式采样"）。
- **`--aug N`（主线 N=8）**：每掩码额外造 N 个合成目标——从可见字段中抽 2–4 个做逐行乘积并标准化（`synth_targets`），拼到 Y 的 conf 维后一起进闭式解与损失（12+8=20 列）。作用是正则化（意外使真实任务也提升）。
- **优化器/早停**：Adam lr=1e-3, weight_decay=5e-4，grad-clip 5.0；train 内部 15% 作 val（seed=args.seed 的置换）；**val 任务固定**（RandomState(999) 生成的 8 个掩码 + val 行对半 support/query），每 epoch 结束算一次闭式 val loss，PATIENCE=60 无提升即停，回滚 best state。
- **实测**：4589 训练行 ×44 字段；45–234s 单卡收敛（RTX 4090）；ckpt 633KB（≈15.6 万参数）。

**查询端（`featridge.ridge_r2`，估计 v_c(S)）**：批量 (B 个集合 × 12 conf) 闭式岭回归。alpha 网格 `ALPHAS=(1e-3,1e-2,1e-1,1,10,100)`，train 内部 85/15 fit/val（`VAL_SEED=20260724` 固定置换）逐 (S,c) 独立选 alpha，再用全 train 重拟合、在 audit 分片上算 out-of-sample R²，`clamp_min(0)`。`ret_resid=True` 时额外返回逐样本残差平方（FDR 用）。特征组装 `build_features` 支持 `last/cat3/poly2/last+p2/cat3+p2`（poly2=集合内字段 raw+平方+两两乘积）。实测 2.2–4.2ms/集合（含 solve）。

---

## 3. DNN 重训认证协议（`scripts/certify_highorder.py`）

- **每个候选集合 S（|S|=m）重训 1+m 个专用 DNN**：S 自身 + 全部 m−1 阶子集（o5 是 6 个，o4 是 5 个）。
- **专用 DNN 架构**（`certify_highorder.DNN`，逐字沿用 88 号修正版 worker）：`Linear(|S|,128)-ReLU-Dropout(0.15)-Linear(128,128)-ReLU-Dropout(0.15)-Linear(128,12)`；输入只有该子集的列（标准化值，无掩码位）。
- **训练**（`train_generic`）：Adam lr=1e-3, wd=5e-4，batch=128，EPOCHS=400，**PATIENCE=120**；**★88 号修正的关键点：早停 validation 取自训练集内部 15%**（固定 `torch.Generator().manual_seed(20260725)` 置换），**测试集完全不参与训练/早停**（67/68 号旧 worker 曾用测试集早停，是被 88 号查出并修正的漏洞，绝对值偏乐观）；val MSE 无提升 120 epoch 即停并回滚 best。
- **评估**：best 模型在全 test 上前向一次，分别在 audit 分片（`SPLIT_SEED=880725` 置换的后一半，与结构化扫描同口径）和全 test 上算逐 conf R²（`per_conf_r2`，clamp≥0）。
- **认证统计量**：`syn_true_audit = max_c [ v(S)_c − max_T v(T)_c ]`（audit 口径为准；full-test 口径并列输出供与 67/68/77 旧真值可比）。另记 `syn_at_struct_conf`（在结构化查询选的那个 conf 上的增量）与 `maxsub_true_audit`（用于"子集反超"诊断：maxsub>v(S) 即根本无增量）。
- **判定标准**：94 号分析统一用 **syn_true_audit > 0.10 = "真协同"**；组间用 Mann-Whitney U 单边检验（`analyze_certify.py`）。对照组 = `syn<0.05` 中 `PICK_SEED=930726` 随机抽样。
- **重训随机性**：每次 v_of 前 `torch.manual_seed(args.seed); np.random.seed(args.seed)`（默认 0）；数据切分本身固定 seed 42。
- **耗时证据**（RUNLOG.md + CHANGELOG）：单个专用 DNN ≈26–31s；**单集合（1+4/1+5 个模型）≈160–186s**（4 分片 ×15 集合用时 2372–2793s；94 号成本表记 "~160–180 s/集合"）。o5 top-21 认证共 126 次重训 ≈63 GPU·min。

---

## 4. 跨 seed 集成过滤（94 号 `ensemble_filter.py` / `seed_agreement.py`）

- **seed 数：3**（L1 checkpoints `oracle_l1_aug8_seed{0,1,2}.pt`；seed 影响主干初始化、掩码流、val 划分，全独立训练）。
- **输入**：seed0 的候选 + 认证结果（`certify_o4_l1top_s*.csv` 的 strong 组，20 个），以及 seed1/seed2 的 **o4 全扫** `scan_o4_oracle_l1_aug8_seed{1,2}_last_s*of*.parquet`（RUNLOG：seed1/2 各先建 subs_o3 缓存 ~37s，再 395–401s 扫完 135,751 个四元组）。
- **一致性统计量**：把 seed1/seed2 各自的全扫按 syn 降序得排名 `rank_s1/rank_s2`（1-based），**`cross_rank = √(rank_s1 · rank_s2)`（几何平均排名）**。
- **阈值：`cross_rank ≤ 500`**（300–1000 都测过，300–500 结果相同）。真协同（认证>0.10 的 9 个）跨 seed 排名中位 ~105，假候选中位 ~27,647，**分离 263×**；过滤后精确率 45%→100%，召回 9/9。
- **诚实标注**（须带到 CAISO）：阈值是在同一批 20 个认证集合上选的，需前瞻验证（预注册阈值→新候选→再认证）。
- `seed_agreement.py` 另做三件事：A) 各 seed 的 o2 质量汇总（读 `eval_l1_order2_aug8{,s1,s2}.csv` 的 last 行）；B) o4 全扫跨 seed 全体 Spearman（0.62/0.66）与 top-208 重叠率；C) seed0 已认证真协同在其他 seed 扫描中的排名。
- **方法论要点**：这与 84 号"同族集成 σ 失明"不矛盾——这里是**独立重训的秩一致性**（打破共享摊销偏差），不是同一模型集成的方差。

---

## 5. 数据集特定硬编码（CAISO 必改清单）

**数据形状（PJM 现状）**：原始 CSV 8760 行 → dropna 后 6556 行；数值列经 drop 后 = **44 general + 12 confidential**；shuffle 切分 seed=42、train_ratio=0.7 → train 4589 / test 1967；test 再按 `SPLIT_SEED=880725` 对半 → search 983 / audit 984。**44 和 12 都不是代码硬编码**——`prepare_data` 动态计算（general = 全部数值列减去 confidential 减去 drop 列），44/12 只出现在注释与组合数里。

**必改点（逐文件）**：
1. **`base.yaml`**（93/94 相同）：`dataset.csv_path`、`fields.drop_columns`（PJM 的 11 个时间戳/泄露列）、`fields.confidential`（PJM 的 12 个机密字段名）——**全部是 PJM 列名，CAISO 需整段重写**。其余可保留：`split_mode: shuffle`、`training.train_ratio: 0.7`、`runtime.seed: 42`（model/graph/correlation 段是历史遗留，本管线不用）。
2. **csv_path 硬编码覆盖**（base.yaml 里的路径实际被覆盖，改 yaml 不够！）：`train_l1.py:111`、`scan_order5.py:57`、`certify_highorder.py:137`、`paired_fdr.py:82`、`eval_l1_order2.py:89`、`eval_l1_order3.py:81`、`inject_l1.py:101`、`l0_on_certified.py:86` —— 全是同一行 `cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")`。
3. **checkpoint 常量**：`paired_fdr.py:49`、`eval_l1_order2.py:41`、`eval_l1_order3.py:38`、`inject_l1.py:55`、`l0_on_certified.py:42` 的 `CKPT = REPO/"DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt"`（CAISO 上需先训一个新的 75 号式 oracle 或直接指向 CAISO L1 ckpt；`paired_fdr.py` **没有 --ckpt 参数**，必须改代码）。
4. **PJM 真值表**（CAISO 上不存在，相关评估脚本需自建真值或跳过）：`eval_l1_order2.py:42`（68 号 `synergy2_perconf.csv`，列 i/j/conf/synergy，conf 为**字段名字符串**，按 `data["confidential"]` 顺序映射）、`eval_l1_order3.py:39`（77 号 `h2_unbiased_pool.csv`）。
5. **90 号 full 字典扫描**：`certify_highorder.py:115` 默认读 `DNN_Aggresvation90/outputs/rescan_o{m}_full_audit_s*.parquet`——CAISO 上若要 full 对照组需先复刻 90 号扫描，否则一律传 `--source` 走 L1 自选路径。
6. **跨目录常量**：各脚本的 `R69 = REPO/"DNN_Aggresvation69"`（源码依赖，可保留）；94 号脚本的 `R93/R90` 与 glob 前缀（`scan_o4_oracle_l1_aug8_seed{0,1,2}_last`、`certify_o4_l1top_s*` 等）都随 ckpt 命名变化，需同步改。
7. **audit 样本定义**：不是数据集属性，而是 `featridge.SPLIT_SEED=880725` 对 test 行的置换后一半（`perm[len//2:]`），前一半是 search。**扫描/FDR/认证三处必须用同一 SPLIT_SEED**，换数据集不需要改，但不能各自另设。
8. **归一化**：两层。(a) `prepare_data`→`standardize`：全部列用 **train 均值/标准差** z-score（std=0 替换为 1）；(b) 扫描/FDR 脚本内对 poly2 特征另做一次 train 统计量的标准化（`mu,sd = Xtr_np.mean/std`）；闭式解内部再做逐批特征标准化。全自动，无需改。
9. **固定种子汇总**（保持不变即可复现）：数据切分 42；`SPLIT_SEED=880725`（search/audit）；`VAL_SEED=20260724`（岭回归 alpha 选择的 fit/val）；认证早停划分 20260725；对照组抽样 `PICK_SEED=930726`；L1 val 任务 999、掩码流 1234+seed。
10. **规模敏感项**：字段数 nG 变了后 C(nG,5) 会变（44→1.09M；若 CAISO 字段更多则全枚举成本按 nG⁵ 增长），`scan_order5` 的分片数与 `--chunk` 需相应调整；`MLPOracle` 输入维 2·nG 自动适应。

---

## 6. GPU 使用方式

- **device 指定**：代码内一律 `torch.device("cuda")` 或 `"cuda" if available`（`train_l1.py:46`、`scan_order5.py:56`、`certify_highorder.py:53`、`paired_fdr.py:80`）。**没有 --device 参数、没有 DDP/多卡代码**——"四卡并行"是靠**同一脚本起 4 个进程、`--shard i --nshard 4` 数据分片 + 启动时 `CUDA_VISIBLE_DEVICES=i` 环境变量**实现的（repo 内 grep 不到 CUDA_VISIBLE_DEVICES，launcher 命令未落盘；RUNLOG 中 4 个分片同时刻 START、独立 DONE 是并行证据）。`base.yaml` 的 `runtime.device: cuda:0` 未被这些脚本使用。
- **硬件**：本机 4× RTX 4090（24GB）。
- **显存**：无显式测量日志。模型极小（MLPOracle ~15.6 万参数 633KB ckpt、认证 DNN hidden=128），大头是扫描时的批量特征张量（chunk=64 集合 × 4589 行 × 256 维 float64 ≈ 0.6GB + Gram/solve 缓冲），单卡 24GB 远够；认证/训练阶段占用可忽略（<2GB 量级推断）。
- **各阶段耗时实证**（`DNN_Aggresvation93/RUNLOG.md` + 94 号 CHANGELOG §6 成本表，全实测）：
  - L1 训练：45–234s，单卡，一次性；
  - L1 查询：2.2–4.2 ms/集合（o5 单分片 271,502 集合 / 583–1050s）；
  - o4 子集缓存 294s、o5 四分片扫描 ~10min（四卡）；o4 全扫 ~280s/半片（两卡）；
  - paired FDR order-3：497–629s 单卡（search 侧 ~250–314s，audit 侧再 ~250–330s）;
  - 认证：~160–186 s/集合（1+m 个专用 DNN，每个 26–31s）；o5 三组 60 集合 ×4 分片 ≈40–47min 墙钟；
  - **o5 全空间管线总账：<2 GPU·h**（40 GPU·min 扫描 + 63 min 认证 top-21），对比暴力全认证 ≈424 GPU·天，加速 ≈5600×。

**两个已知坑**（复刻时注意）：① `paired_fdr.py` 的 CKPT 写死且无参数；② `certify_highorder.py` 的断点续跑 glob（`certify_o{m}_s*of*.csv`）不区分 `--seed`/`--source`，同 order 下换来源重跑时若集合重叠会被静默跳过，CAISO 上建议清空旧 CSV 或改 glob 加 tag。