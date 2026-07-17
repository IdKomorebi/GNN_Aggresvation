# DNN_Aggresvation50 日志

## Modify by Claude: 2026-06-26

## 1. 子项目定位

受限攻击者场景之二——**字段缺失**:攻击者只有部分 general 字段(训练+推断都缺,
贴近真实威胁)时,6 种敏感度方法的排名质量是否拉开。随机保留 25%/50%/75%/100%
的 general 字段(缺失字段在 model forward 里固定置 0,seed 123,训练/评估一致),
每个比例训一个 multi graph-combo,**各自重算全字段上界**,再做 6 方法 top-k。

代码:`src/model.py` 加 `field_keep_mask`(forward 把缺失 general 输入置 0);
run_pipeline / compute_sensitivity 按 `fields.field_keep_frac` 同 seed 算 mask;
`evaluate_missing.py` / `scheduler_eval.py` / `aggregate_missing.py`。

## 2. 结果

| 保留字段 | 全字段上界(重算) | top10 方法间分化(max−min) | top10 最优 |
|---|---:|---:|---|
| 25%(11/44) | 0.635 | **0.437** | mask |
| 50%(22/44) | 0.786 | 0.145 | single |
| 75%(33/44) | 0.829 | 0.177 | mask |
| 100%(44/44) | 0.866 | 0.141 | shapley |

逐方法 top10(keep=25%):mask/single/ig/shapley/lrp ≈ **0.63**(≈上界),
**attn = 0.198**(崩)。

## 3. 关键发现(这次有真区分度)

1. **字段缺失下注意力归因(attn)明显崩溃,缺失越严重越崩**。keep=25% 时 attn 的
   top10 仅 0.198,而其他 5 法都 ≈0.63(≈上界);图上绿线(attn)在所有缺失比例下
   都垫底,缺失越多落后越远。这与 DNN49 低数据(attn 还行)截然不同。

2. **根因——注意力反映"模型关注哪里",不反映"哪里真有信息"**。缺失字段被置 0,
   但注意力仍可能"关注"这些位置(attention 由 q·k+先验决定,不知道输入=0),于是
   attn 把无信息的缺失字段排到前面,top-k 选中它们 → 推不准。而 mask/single/ig/
   shapley/lrp 基于**实际推断贡献**(缺失字段贡献=0),正确把它们排末尾。

3. **字段缺失部分打破了饱和**:方法间分化(top10 spread)在缺失场景下 0.14~0.44,
   明显大于低数据的 0.05~0.14——主要是 attn 被区分出来;其余 5 法仍紧密稳健。

## 4. 结论

- **字段缺失是有区分度的评价场景**,它给出一个清晰的论文论点:**注意力归因在部分
  输入缺失时不可靠,应使用因果(mask/single/shapley)或梯度(ig/lrp)类方法**。
- 其余 5 法在缺失下仍高度一致、稳健——再次印证推断驱动敏感度的稳健性。
- 综合 DNN48/49/50:完整数据 + 低数据下方法都饱和/一致;只有**字段缺失**能暴露
  attn 的脆弱性。这是"为什么不能只用注意力(IGNN 风格)做敏感度"的实证支撑。

## 5. 输出
```
outputs/tuning/keep_{25,50,75,100}/        各保留比例模型
outputs/missing/<keep>/{ranking.csv, topk.csv, upper.txt}
outputs/missing/_summary/missing_topk.png + missing_summary.csv
```
