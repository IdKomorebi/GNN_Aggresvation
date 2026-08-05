# DNN_Aggresvation80 运行流水

> 本文件由 `src/runlog.py` 自动追加，与 `CHANGELOG.md`（结论）并列。
> 机器可读版本：`outputs/events.jsonl`

## 2026-07-24
- `02:27:36` ▶ **[BUILD] START**　构造候选排序键
- `02:27:49` ✔ **[BUILD] DONE**　13244 三元组 × 5 K；带真值 2197 个　　进度 None/66220
- `02:29:04` ◆ **[EVAL] DECISION**　最强档 K=0 Top30%：现状 S1=0.047，最优键 S2_vijk=0.811
- `02:30:07` ✔ **[UNION] DONE**　组合排序方案评估完成，见 outputs/union_eval（stdout）
- `02:32:03` ✔ **[DIAG] DONE**　信号上限 + 层级生成覆盖率诊断完成
- `02:33:23` ◆ **[CONCLUDE] DECISION**　问题1机制钉死:syn3与v_ijk排序偏好相反且在最强档交叉(v_ijk百分位58→84单调升,syn3 94→26骤降);K=0免费筛选有内在天花板,最强档并集救到28%
- `02:33:23` ◆ **[CONCLUDE] DECISION**　问题2出路:层级候选生成(强二阶阈值0.1扩展)9841候选覆盖99.3%强三阶,四阶可从强三阶扩展绕开全空间;剥离法只做防护集不做发现,判定正确
