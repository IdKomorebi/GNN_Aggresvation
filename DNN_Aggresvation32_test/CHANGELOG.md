# DNN_Aggresvation32_test 日志

## Modify by Codex: 2026-06-23

### 1. 子项目定位

本子项目是 DNN_Aggresvation32 的窗口消融实验。DNN32 配置为 `split_mode:
shuffle` 且 `window_size: 4`，由于先随机打乱数据再构造窗口，窗口里的 4 个值不再
代表真实连续时间，因此这个实验把所有配置统一改为：

```yaml
model:
  window_size: 1
```

其它核心设置保持 DNN32 同口径：bipartite + shuffle split + nostaged + allloss，
并完整重跑 1 个 multi + 12 个 single。

### 2. 代码与运行改动

- 从 `DNN_Aggresvation32` 复制得到 `DNN_Aggresvation32_test`。
- 13 个配置的 `window_size` 从 4 改为 1。
- `scripts/scheduler.py` 的项目路径指向 `DNN_Aggresvation32_test`。
- 为避免 `dcor/numba` 在当前运行环境中缓存失败，scheduler 给子进程设置
  `NUMBA_CACHE_DIR=/tmp/numba_cache`。
- 相关性缓存从 DNN32 副本复用；窗口长度不影响字段相关性。

### 3. 实验结果汇总

| 版本 | multi mean R2 | multi median R2 | single mean R2 | single median R2 | single-multi mean gap | single wins |
|---|---:|---:|---:|---:|---:|---:|
| DNN32 window=4 | 0.7868 | 0.7938 | 0.8273 | 0.8233 | 0.0405 | 10/12 |
| DNN32_test window=1 | **0.8096** | **0.8182** | **0.8898** | **0.8812** | 0.0802 | 12/12 |

去掉窗口后，multi 和 single 都提升了；single 提升更大，因此 single-multi gap
反而扩大。

### 4. 关键字段变化

| 字段 | DNN32 multi | w1 multi | multi 提升 | DNN32 single | w1 single | single 提升 |
|---|---:|---:|---:|---:|---:|---:|
| congestion_price_da | 0.3012 | **0.4175** | +0.1163 | 0.5171 | **0.8326** | +0.3155 |
| congestion_price_rt | 0.5543 | **0.5667** | +0.0124 | 0.5028 | **0.6040** | +0.1012 |
| net_actual_interchange_mw | 0.7427 | **0.8078** | +0.0650 | 0.8198 | **0.8853** | +0.0655 |
| gross_actual_interchange_mw | 0.7262 | **0.7470** | +0.0207 | 0.7351 | **0.8102** | +0.0750 |
| thirty_minutes_reserve | 0.7972 | **0.8194** | +0.0222 | 0.7876 | **0.8673** | +0.0796 |

完整表见：

- `outputs/multi_vs_single_window1.csv`
- `outputs/window1_vs_DNN32_window4.csv`

### 5. 结论

1. DNN32 的 `shuffle + window_size=4` 确实不合理；窗口不再是时间窗口，反而像给
   每个 General 节点塞了 3 个随机时刻的噪声特征。
2. 去掉窗口后，整体效果更好：multi 平均 R2 +0.0229，single 平均 R2 +0.0626。
3. 改善最大的是价格类和交换类字段，尤其 `congestion_price_da`：single 从 0.517
   跳到 0.833，multi 从 0.301 到 0.417。
4. single 提升幅度大于 multi，说明“干净输入”会让单目标上限更高，同时也把
   multi 的共享表示瓶颈暴露得更明显。

后续如果还想利用真实时间信息，应改为“先按原始时间构造连续窗口，再对样本做
shuffle split”，不能像 DNN32 那样先 shuffle 再窗口。
