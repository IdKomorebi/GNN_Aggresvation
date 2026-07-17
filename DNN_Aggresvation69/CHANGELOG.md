# DNN_Aggresvation69 日志

## 1. 子项目定位

本项目把“GNN是否更适合随机失活和间接推断”拆成两个不能混淆的问题：

1. **专用攻击者上限**：固定字段集合后从头训练的 DNN 与 GCN，谁能推断出更高的测试集 R²？
2. **通用随机失活 oracle 保真度**：MLP-oracle 与 GNN-oracle 对任意掩码集合的估计，谁更接近
   专用重训上限？相同步数微调后谁收敛更快？

并且分别评估：

- 直接集合敏感度 `v_c(S)`；
- 二阶协同 `v_c(ij)-max(v_c(i),v_c(j))`；
- 三阶不可约协同 `v_c(ijk)-max(v_c(ij),v_c(ik),v_c(jk))`。

## 2. 数据与公平性设计

- 60号：50个宽尺寸随机集合，直接复用已有 DNN/GCN 重训。
- 67号：44个单字段 + 946个字段对，复用全部 DNN 重训；GCN 分层审计9个字段对。
- 68号：197个候选三元组 + 200个随机三元组，复用全部 DNN 重训；GCN 分层审计9个三元组。
- 共1437个评测条目、1431个不同字段集合；其中60号随机抽样有6个条目恰好与67号穷举集合
  重合。这6组保留为独立重训重复验证，但总体结论按 `canonical_sid` 防止误报为不同集合。
  统一数据切分 seed=42、重训 seed=0。
- 全量集合/协同认证使用现成 DNN 重训真值；有GCN结果的50个宽尺寸集合和18个分层审计集合，
  额外报告逐机密字段 `max(DNN, GCN)`。两种口径分开，禁止把缺少GCN的集合伪称 best-of-struct。
- 两个通用 oracle 使用相同 mask、batch 顺序、Adam、学习率及 K={0,10,50,200}。选择200步作为
  全量终点，是因为两种结构的同集合GPU小样显示MLP 500步约2秒、GNN约12秒；200步足以看清
  同步数收敛趋势，同时避免把结构差异误写成不等计算预算。
- K=0 另做三个 oracle 种子的稳定性检查。

## 3. 产出约定

运行完成后：

```
RESULTS.md
outputs/source_manifest.json
outputs/attacker_upper_bound.csv
outputs/attacker_upper_bound_per_conf.csv
outputs/oracle_fidelity.csv
outputs/oracle_fidelity_per_conf.csv
outputs/architecture_comparison.csv
outputs/architecture_per_conf_comparison.csv
outputs/synergy_metrics.csv
outputs/synergy_metrics_per_conf.csv
outputs/bootstrap_architecture_comparison.csv
outputs/k0_seed_robustness.csv
outputs/overview.png
outputs/fidelity_by_size.png
```

## 4. 结论

1. **先区分GCN的两个角色。** 固定集合后从头训练时，GCN在5--8、9--16字段段的平均R²
   分别比DNN高0.0235、0.0130，说明图结构对中等大小集合的专用攻击者确实有帮助；但在9个
   字段对和9个三元组审计样本上，GCN平均分别低0.0378、0.0290。因此不能笼统地说
   “GCN上限更高”，更准确的说法是它在部分尺寸、部分机密目标上补强攻击上限。
2. **通用随机失活模型仍然是MLP更准。** 不微调时，对专用DNN重训的逐目标MAE，MLP/GNN
   在宽尺寸集合为0.0767/0.1362，在字段对为0.0495/0.0813，在三元组为0.0912/0.1381。
   三个oracle种子的复核保持同一结论，因而不是单个checkpoint偶然造成。
3. **同等200步微调以后，两者都明显改善，但GNN没有追上。** MLP/GNN的MAE在宽尺寸集合
   降到0.0194/0.0784，字段对降到0.0150/0.0350，三元组降到0.0213/0.0436；同时MLP
   每集合约0.39--0.40秒，GNN约3.07--3.11秒。这里比较的是同等优化步数，不是同等时长。
4. **GCN没有让协同更清楚。** 对946个二阶协同，K=0时MLP/GNN与重训协同排序的
   Spearman为0.6708/0.3213，K=200为0.9279/0.5309；对397个三阶候选则由
   0.4766/0.1418提升到0.9183/0.5410。GNN在K=0、10的二阶top-20命中率有小幅优势，
   但全局排序和绝对误差都更差，K=50、200后MLP也追平并反超top候选命中。
5. **本号的决策。** GCN适合作为“专用攻击者结构之一”，用于把部分字段集合的经验攻击
   上限做强；不适合作为当前通用oracle的默认结构。通用估计继续采用MLP，并用少量微调闭合
   其系统性低估。当前静态相关图、邻居筛选和消息聚合没有转化成更好的高阶协同保真度。

三元组 `triple_top` 来自早期MLP扫描，存在选择偏差，因此三阶结构比较同时保留了200个
随机三元组；GCN字段对/三元组结果是各9个的分层审计，不应外推成全部小集合的穷举结论。

完整表格、逐机密字段结果、bootstrap结构差异、三种子稳定性和图见 `RESULTS.md` 与
`outputs/`；`outputs/validation_checks.json` 记录了文件完整性、字段一致性、重复集合一致性
以及69号对67号既有结果的精确复现检查。
