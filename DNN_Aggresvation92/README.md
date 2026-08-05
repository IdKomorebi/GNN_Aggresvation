# DNN_Aggresvation92

本号验证一个可证伪的问题：

> 通用 MLP oracle 对不同字段集合做 K=25 全参微调时，有效参数变化是否落在一个
> 可迁移的低维子空间；若是，能否只用当前集合的一次梯度，预测出接近 K=25 的更新？

这不是把 LoRA 换个名字再跑一次。实验包含三个彼此分开的层次：

1. **表示上限**：审计任务自己的 K25 更新投影进“仅由训练任务学习”的子空间后，
   能保留多少性能。它不可部署，只回答低维假设是否成立。
2. **一次梯度可达性**：只给模型当前集合的一批训练样本和一次反向传播，
   能否从梯度预测低维更新系数。
3. **PEFT 对照**：标准 LoRA 在同样 K=1/5/25 更新预算下是否更好。

训练、validation、audit 按集合严格拆分。对于 validation/audit 三元组，其全部二元父集
也从子空间训练任务中排除，以免协同指标发生集合泄漏。

运行顺序：

```bash
conda run -n Pytorch310_codex python scripts/build_design.py
CUDA_VISIBLE_DEVICES=1 conda run -n Pytorch310_codex python scripts/collect_trajectories.py --shard 0 --nshard 3
CUDA_VISIBLE_DEVICES=2 conda run -n Pytorch310_codex python scripts/collect_trajectories.py --shard 1 --nshard 3
CUDA_VISIBLE_DEVICES=3 conda run -n Pytorch310_codex python scripts/collect_trajectories.py --shard 2 --nshard 3
CUDA_VISIBLE_DEVICES=1 conda run -n Pytorch310_codex python scripts/analyze_subspace.py
```

完整结论见 `CHANGELOG.md`。
