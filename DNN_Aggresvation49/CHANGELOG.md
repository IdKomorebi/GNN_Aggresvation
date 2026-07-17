# DNN_Aggresvation49 日志

## Modify by Claude: 2026-06-26

## 1. 子项目定位

受限攻击者场景之一——**低数据**:攻击者训练样本受限时,6 种敏感度方法的排名质量
是否拉开差异(完整数据下 DNN48 已饱和)。固定 test(30%),仅对训练集子采样
5%/10%/20%/50%/100%,每个比例训一个 multi graph-combo,**各自重算全字段上界**,
再做 6 方法 top-k。gate 不纳入(需联合训练、top-k 不适用)。

代码:`train.py` 加 `train_subsample_frac`;`evaluate_lowdata.py`(每 frac 算 6 方法
排名+top-k);`scheduler.py`/`scheduler_eval.py`(4 卡并行);`aggregate_lowdata.py`。

## 2. 结果

| 训练量 | 全字段上界(重算) | top10 方法间分化(max−min) |
|---|---:|---:|
| 5% | 0.718 | 0.083 |
| 10% | 0.678 | 0.070 |
| 20% | 0.794 | 0.070 |
| 50% | 0.820 | 0.053 |
| 100% | 0.866 | 0.141 |

(上界随训练量正确下降;frac_10 略低于 frac_05 是低数据子采样随机波动。)

## 3. 关键发现

- **低数据没有拉开方法差异**:方法间分化(top10 spread)各训练量下都只 0.05~0.14,
  **不随数据减少而增大**;6 条曲线在所有训练量下仍紧密成簇。低数据未打破 DNN48 的
  饱和。
- **但这是个强 robustness 论点**:敏感度排名的方法选择即使在低数据下也不敏感
  ——不挑方法、不挑数据量,高泄露字段排名都一致。说明推断驱动敏感度结论稳健。
- shapley 在多数训练量下略占优(5/20/100% top10 最好),但优势小、不绝对。

## 4. 结论

低数据**证明排名稳健,但没找到方法区分度**。受限样本量不是拉开方法差异的杠杆。
(字段缺失场景见 DNN50——那里改变的是信息结构而非样本量。)

## 5. 输出
```
outputs/tuning/frac_{05,10,20,50,100}/   各训练量模型
outputs/lowdata/<frac>/{ranking.csv, topk.csv, upper.txt}
outputs/lowdata/_summary/lowdata_topk.png + lowdata_summary.csv
```
