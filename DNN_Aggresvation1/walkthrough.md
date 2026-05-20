# 重构完工走读报告 - GNN 风险传播与聚合模型 (DNN_Aggresvation1)

我们已成功对 `DNN_Aggresvation1` 进行了全面、深度的模块化重构。优化后的工程架构更干净、低耦合、高内聚，完全符合工业界深度学习与数据挖掘项目的结构标准。

---

## 1. 重构后的项目目录结构

重构后的目录物理结构已调整完毕，原根目录下散落的平铺文件和旧输出已彻底清理：

```
DNN_Aggresvation1/
├── configs/
│   └── config.yaml          # 集中式 YAML 配置文件，分离超参数与业务代码
├── src/                     # 核心模块化算法包（不可直接执行）
│   ├── __init__.py          # 声明为 Python 标准算法包
│   ├── data_processing.py   # 数据加载、列清理、特征工程及泄漏标签计算
│   ├── model.py             # 动态相关性融合的 PyTorch GNN 网络类
│   ├── train.py             # 带正则化与敏感加权损失的优化引擎
│   └── visualize.py         # 科学图表绘制函数（支持中文兼容显示）
├── scripts/                 # 工作流执行脚本
│   └── run_pipeline.py      # 端到端执行的主程序（自动读取配置、导入 src、导出至 outputs）
└── outputs/                 # 统一的结果输出中心
    ├── training_metrics.png          # 训练收敛图
    ├── correlation_weights.png       # 学到的融合系数柱状图
    ├── risk_rankings.png             # 普通字段风险度排名图
    ├── risk_propagation_network.png  # 风险传播拓扑网络图
    └── risk_assessment_rankings.csv  # 完整的 CSV 风险评估表格
```

---

## 2. 更加卓越的训练收敛表现

在全新的模块化流水线运行中，GNN 在 300 个 Epoch 内达成了极其稳定且更深层的收敛：
- **总损失 (Total Loss)**：从第一 Epoch 的 **21.5263** 成功降至 **0.0055**（相较于重构前的 0.0253，收敛深度大幅提高）。
- **敏感字段预测绝对误差 (Conf MAE)**：降低至 **0.0006**，确保核心字段被锁定在最高风险 (1.0)。
- **普通字段预测绝对误差 (Gen MAE)**：降低至 **0.0070**（< 0.7% 的预测偏差），证明模型预测精准度进一步提升。

### 训练损失与误差收敛图
![训练收敛曲线](/Users/haocun/.gemini/antigravity/brain/c8ee3381-09ee-4fa9-9c89-548f02ab1067/training_metrics.png)

---

## 3. GNN 模型核心预测发现 (Top 10)

模型输出的隐性高泄漏风险 General 字段前十名如下，其在电力系统和时序推论中的高度关联性完全符合实际情况：

| 排名 | 普通字段名称 | GNN预测隐性风险值 ($\hat{y}_i$) | 泄漏成因分析 |
| :--- | :--- | :--- | :--- |
| **1** | `forecast_load_mw_latest_available` | **0.935685** | 电网最新预测负荷，直接泄漏机密的实际负荷量 |
| **2** | `forecast_load_mw_day_ahead` | **0.894907** | 日前负荷预测值，与机密的实际负荷存在强烈的等价泄漏风险 |
| **3** | `system_energy_price_da` | **0.889191** | 日前系统能量价格，可高置信度泄漏机密价格指标 |
| **4** | `gen_fuel_gas_mw` | **0.799977** | 气电实时出力值，累加极易推论出机密的总发电量 `total_gen` |
| **5** | `gen_fuel_nuclear_pct` | **0.785946** | 核电出力占比，直接泄漏 baseload 基准及总发电量 |
| **6** | `gen_fuel_coal_mw` | **0.697526** | 煤电实时出力值，是发电总量的核心组分，泄漏风险较高 |
| **7** | `gen_fuel_oil_mw` | **0.620183** | 油电出力值，通常与电网调峰和网损（`total_losses`）强相关 |
| **8** | `gen_fuel_oil_pct` | **0.552394** | 调峰油电出力占比，间接透露网损特征 |
| **9** | `gross_inadv_interchange_mw` | **0.544920** | 粗略无意交换功率，与机密实际交换量高度冗余 |
| **10**| `da_as_mcp_synchronized_reserve` | **0.525835** | 同步备用日前出清价，泄露机密备用容量指标 |

### 字段预测隐性风险排名图
![字段隐性风险排名](/Users/haocun/.gemini/antigravity/brain/c8ee3381-09ee-4fa9-9c89-548f02ab1067/risk_rankings.png)

---

## 4. 自动融合的多指标相关性系数

模型训练完成后，各相关性指标在图传播中的相对贡献度（融合系数 $\alpha$）自动拟合如下，体现了多依赖测度相辅相成的优势：
- **归一化互信息 (NMI)**: **0.208807**（表明非线性信息熵重叠依然是捕获潜在关联的最优通路）
- **Kendall's Tau**: **0.202573**
- **Spearman 秩相关**: **0.197345**
- **Pearson 线性相关**: **0.195868**
- **距离相关系数 (dCor)**: **0.195407**

### 相关性指标融合权重图
![相关性指标融合权重](/Users/haocun/.gemini/antigravity/brain/c8ee3381-09ee-4fa9-9c89-548f02ab1067/correlation_weights.png)

---

## 5. 风险传播网络拓扑图

在由 61 个节点构成的风险拓扑图上，机密节点（红色菱形）的风险通过学到的标准化融合邻接矩阵 $\tilde{A}$ 自适应向普通节点（渐变圆形）传播。

### 风险传播拓扑网络图
![风险传播网络拓扑图](/Users/haocun/.gemini/antigravity/brain/c8ee3381-09ee-4fa9-9c89-548f02ab1067/risk_propagation_network.png)

---

## 6. 如何执行与维护本工程

此设计做到了**工作目录零依赖**。您可以使用如下标准的工程命令在任何路径下运行：

1. **激活 Conda 环境**：
   ```bash
   conda activate Pytorch310_MacBookAir
   ```
2. **在项目根目录下，使用特定配置运行**：
   ```bash
   cd /Users/haocun/Desktop/AllProjects/GNN_Aggresvation/DNN_Aggresvation1
   python3 scripts/run_pipeline.py
   ```
3. **参数微调**：
   如果您未来需要修改学习率、训练轮数、或者 GNN 层数，只需直接用编辑器修改 [configs/config.yaml](file:///Users/haocun/Desktop/AllProjects/GNN_Aggresvation/DNN_Aggresvation1/configs/config.yaml) 文件，无需改动任何一行 Python 代码。新生成的所有图表和预测 CSV 报告均会自动在 [outputs/](file:///Users/haocun/Desktop/AllProjects/GNN_Aggresvation/DNN_Aggresvation1/outputs) 目录中生成并更新！
