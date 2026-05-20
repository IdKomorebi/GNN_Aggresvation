import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNRiskAggregator(nn.Module):
    def __init__(self, in_features=14, hidden_dim=64, num_layers=2, alpha_temperature=1.0):
        """
        可微分 GNN 风险传播与聚合模型。
        - in_features: 节点 raw 特征维度（1 维敏感性初值 + 13 维标准化统计特征）
        - hidden_dim: 隐层维度
        - num_layers: 空间信息聚合层数 (L)
        """
        super(GNNRiskAggregator, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.alpha_temperature = max(float(alpha_temperature), 1e-6)
        
        # 1. 5种相关性矩阵的学融合参数 beta，初始化为全 0（等价于初始权重均为 0.2）
        self.beta = nn.Parameter(torch.zeros(5, dtype=torch.float32))
        
        # 2. 输入映射投影层: h_i^(0) = ReLU(W_in * h_raw + b_in)
        self.in_proj = nn.Linear(in_features, hidden_dim)
        
        # 3. GNN 空间信息传递层
        self.W_self = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.W_nei = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])
        
        # 4. 输出映射层: y_hat = Sigmoid(W_out * h^(L) + b_out)
        self.out_proj = nn.Linear(hidden_dim, 1)
        
    def get_correlation_weights(self):
        """
        获取经过 Softmax 归一化后的相关性指标融合权重 alpha_k。
        """
        alpha = F.softmax(self.beta / self.alpha_temperature, dim=0)
        return alpha
        
    def forward(self, h_raw, R, m, epsilon=1e-8):
        """
        前向风险传播计算。
        - h_raw: 原始节点特征，形状 [N, in_features]
        - R: 多相关性矩阵集合，形状 [5, N, N]
        - m: 候选边掩码，形状 [N, N]
        """
        N = h_raw.shape[0]
        
        # 1. 动态融合边权：A_ij = m_ij * sum_k(alpha_k * R_k_ij)
        alpha = self.get_correlation_weights()
        A_comb = torch.sum(alpha.view(5, 1, 1) * R, dim=0)
        A = m * A_comb
        
        # 2. 行归一化 (Row Normalization)
        row_sums = torch.sum(A, dim=1, keepdim=True)
        A_tilde = A / (row_sums + epsilon)
        
        # 3. 特征投影映射
        h = F.relu(self.in_proj(h_raw))
        
        # 4. GNN 空间聚合传播
        for l in range(self.num_layers):
            # 聚合邻居信息: m_l = A_tilde * h
            m_l = torch.matmul(A_tilde, h)
            
            # 更新节点表示: h = ReLU(W_self * h + W_nei * m_l + b)
            h = F.relu(self.W_self[l](h) + self.W_nei[l](m_l) + self.biases[l])
            
        # 5. 输出 Sigmoid 预测概率风险分数
        y_hat = torch.sigmoid(self.out_proj(h)).squeeze(-1)
        
        return y_hat, alpha, A_tilde
