import torch
import torch.optim as optim
import numpy as np

# 引入同一包内的 GNN 风险模型模型类
from .model import GNNRiskAggregator

def train_gnn_model(data_dict, hidden_dim=64, num_layers=2, lr=0.01, epochs=300, 
                     w_c=5.0, lmbda=0.0, seed=None, beta_lr=None,
                     alpha_temperature=1.0, verbose=True):
    """
    驱动 GNN 风险传播模型的完整训练与优化过程。
    - data_dict: prepare_pipeline_data 生成的包含特征、相关性矩阵及监督标签的字典
    - w_c: Confidential 节点损失惩罚加权系数
    - lmbda: 作用于融合参数 beta 的 L2 正则项系数
    """
    
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    # 1. 数据结构转为张量
    h_raw = torch.tensor(data_dict["h_raw"], dtype=torch.float32)
    R = torch.tensor(data_dict["R"], dtype=torch.float32)
    m = torch.tensor(data_dict["m"], dtype=torch.float32)
    y = torch.tensor(data_dict["y"], dtype=torch.float32)
    s_init = torch.tensor(data_dict["s_init"], dtype=torch.float32)
    supervision_mask = torch.tensor(
        data_dict.get("supervision_mask", np.ones_like(data_dict["y"])),
        dtype=torch.float32
    )
    
    N = h_raw.shape[0]
    
    # 2. 建立半监督掩码，并按字段组平衡损失，避免少量 General 锚点被数量/权重淹没
    conf_supervised = ((s_init == 1.0) & (supervision_mask > 0.0))
    gen_supervised = ((s_init == 0.0) & (supervision_mask > 0.0))
    
    # 3. 实例模型与 Adam 优化器
    model = GNNRiskAggregator(
        in_features=h_raw.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha_temperature=alpha_temperature
    )
    if beta_lr is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        beta_params = [model.beta]
        other_params = [p for name, p in model.named_parameters() if name != "beta"]
        optimizer = optim.Adam([
            {"params": other_params, "lr": lr},
            {"params": beta_params, "lr": beta_lr}
        ])
    
    # 训练历史日志数据
    history = {
        "loss": [],
        "weighted_mse": [],
        "conf_mae": [],
        "gen_mae": [],
        "alpha_weights": []
    }
    
    if verbose:
        print("\n" + "="*50)
        print("          开始 GNN 风险模型训练与优化")
        print("="*50)
        print(f"图节点 (列总数):     {N}")
        print(f"Confidential 字段数: {int(s_init.sum().item())}")
        print(f"General 字段数:      {int(N - s_init.sum().item())}")
        print(f"监督 General 标签数: {int(gen_supervised.sum().item())}")
        print(f"隐藏特征维度:        {hidden_dim}")
        print(f"聚合传播层数 (L):    {num_layers}")
        print(f"超参数设置:          学习率={lr}, beta_lr={beta_lr}, "
              f"temperature={alpha_temperature}, w_c={w_c}, lambda={lmbda}")
        print("-"*50)
        
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # 前向传播
        y_hat, alpha, A_tilde = model(h_raw, R, m)
        
        # 分组 MSE 损失：Confidential 保持高风险锚定，General 伪标签锚点独立约束风险刻度
        zero_loss = y_hat.new_tensor(0.0)
        conf_loss = torch.mean((y_hat[conf_supervised] - y[conf_supervised])**2) if conf_supervised.any().item() else zero_loss
        gen_loss = torch.mean((y_hat[gen_supervised] - y[gen_supervised])**2) if gen_supervised.any().item() else zero_loss
        mse_loss = w_c * conf_loss + gen_loss
        
        # 参数 beta 的 L2 正则化项惩罚
        reg_loss = lmbda * torch.sum(model.beta ** 2)
        
        total_loss = mse_loss + reg_loss
        
        # 反向传播并更新
        total_loss.backward()
        optimizer.step()
        
        # 训练指标测度
        with torch.no_grad():
            model.eval()
            eval_y_hat, eval_alpha, _ = model(h_raw, R, m)
            
            # 敏感节点的平均绝对误差
            conf_mae = torch.mean(torch.abs(eval_y_hat[conf_supervised] - y[conf_supervised])).item()
            
            # 少量已标注普通节点的平均绝对误差
            if gen_supervised.sum().item() > 0:
                gen_mae = torch.mean(torch.abs(eval_y_hat[gen_supervised] - y[gen_supervised])).item()
            else:
                gen_mae = 0.0
            
            model.train()
            
        # 记录日志
        history["loss"].append(total_loss.item())
        history["weighted_mse"].append(mse_loss.item())
        history["conf_mae"].append(conf_mae)
        history["gen_mae"].append(gen_mae)
        history["alpha_weights"].append(eval_alpha.cpu().numpy().tolist())
        
        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            alpha_str = ", ".join([f"{a:.4f}" for a in eval_alpha.tolist()])
            print(f"Epoch {epoch:3d}/{epochs:3d} | Loss: {total_loss.item():.6f} | "
                  f"Conf MAE: {conf_mae:.4f} | Gen MAE: {gen_mae:.4f} | "
                  f"融合权重 [Pearson, Spearman, Kendall, NMI, dCor]: [{alpha_str}]")
            
    if verbose:
        print("-"*50)
        print("GNN 模型收敛训练顺利完成！")
        print("="*50 + "\n")
        
    model.eval()
    with torch.no_grad():
        final_y_hat, final_alpha, final_A_tilde = model(h_raw, R, m)
        
    return {
        "model": model,
        "history": history,
        "y_pred": final_y_hat.numpy(),
        "alpha": final_alpha.numpy(),
        "A_tilde": final_A_tilde.numpy()
    }
