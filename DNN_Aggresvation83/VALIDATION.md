# DNN83 验证报告

## 总体判断：可以作为下一阶段依据，但需带 caveat

D83 的“结构化闭式低阶攻击者值得进入主线”结论证据充分；“它已经解决任意高阶集合”
不成立。CVaR 与通用 poly2 oracle 的负结果也可分享。

## 高影响问题

1. **D82 真值字段错误（已在 D83 绕开）**
   - `syn2_true_max` 是二阶协同增量，不是 pair R²；
   - 污染 D82 对新增 1800 点的 parent/pair MAE 和 bias；
   - 不污染 `syn3_true`、强弱标签及只依赖它们的 recall/Spearman；
   - D83 从原始 JSON 重建真值，并用修正真值重算 K-grid。

2. **“重训真值”不是数学 sup**
   - DNN、poly2、arith 都只是攻击者下界；
   - D83 另报 portfolio 最大值，不再把单 MLP seed 当绝对上限。

3. **外推范围**
   - 仅 PJM、一个 train/test 切分；
   - 结构化内部 alpha 选择目前一个固定训练内切分；
   - 高阶字典与缓存充分统计量尚未实现。

## 关键复算

- 修正真值后 uniform K0 parent MAE：`0.0718`；
- K25：`0.0414`，K50：`0.0327`；旧文件中“微调让 parent MAE 上升”的结论来自错误真值；
- 全 13,244 三元组 global Top-30% recall：
  - uniform K0 `0.8424`（seed 0）；
  - uniform K50 `0.9110`；
  - poly2 ridge `0.9630`；
  - arithmetic ridge `0.9384`；
- portfolio 检查：隔离 test 中旧/新强三阶名单 `130/130` 完全重合，分数
  Spearman `0.9991`。

## 泄漏检查

- 所有最终 R² 在原项目隔离 test 上计算；
- ridge alpha 只用训练集内部 fit/validation 选择；
- 特征均值、方差、ratio 裁剪阈值只由训练数据产生；
- test 未参与方法或超参数选择；
- parameter-matched 对照让 rawwide 参数量略多于 poly2，对照不偏向新方法。

## 剩余 caveat

- 结构化模型与 DNN 重训使用同一 test，portfolio 是更强攻击下界，不是独立无偏真值；
- global recall 的阈值为全空间固定保留 30%，可与 D82 比较，但不能代表其他保留率；
- 强协同样本仍稀少，应补 recall–cost 曲线和 bootstrap/Wilson 区间；
- 闭式方法当前每个集合仍重复扫描样本；充分统计量缓存实现后才能评价真正高阶吞吐。
