import os
import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 绘图支持中文显示和负号正常显示
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history, save_dir):
    """
    绘制 GNN 训练收敛图：展示总 Loss、加权 MSE 以及敏感/普通节点的 MAE。
    """
    epochs = len(history["loss"])
    epoch_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. 损失值收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, history["loss"], label="Total Loss", color="#3f51b5", linewidth=2)
    plt.plot(epoch_range, history["weighted_mse"], label="Weighted MSE", color="#e91e63", linewidth=1.5, linestyle="--")
    plt.title("GNN 训练损失收敛曲线", fontsize=12, fontweight="bold")
    plt.xlabel("训练 Epoch", fontsize=10)
    plt.ylabel("Loss 损失值", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    # 2. MAE 误差收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, history["conf_mae"], label="Confidential MAE", color="#f44336", linewidth=2)
    plt.plot(epoch_range, history["gen_mae"], label="Labeled General MAE", color="#4caf50", linewidth=2)
    plt.title("平均绝对误差 (MAE) 曲线", fontsize=12, fontweight="bold")
    plt.xlabel("训练 Epoch", fontsize=10)
    plt.ylabel("MAE 误差值", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 已将训练收敛曲线保存至: {plot_path}")

def plot_correlation_weights(alpha, save_dir):
    """
    绘制 5 种相关性测度的学权重 alpha 柱状图。
    """
    metrics = ["Pearson", "Spearman", "Kendall's Tau", "NMI", "Distance Corr (dCor)"]
    colors = ["#4caf50", "#2196f3", "#9c27b0", "#ff9800", "#e91e63"]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, alpha, color=colors, edgecolor="black", alpha=0.85, width=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    plt.title("融合模型学到的多相关性权重指标分配 (alpha)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("融合权重值 (Softmax 概率值)", fontsize=11)
    plt.ylim(0, max(alpha) + 0.08)
    plt.grid(axis='y', linestyle=":", alpha=0.6)
    
    plot_path = os.path.join(save_dir, "correlation_weights.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 已将学习到的相关性权重图保存至: {plot_path}")

def plot_risk_rankings(columns, y_pred, s_init, save_dir):
    """
    绘制普通字段 (General Columns) 预测风险隐性排名的水平条形图。
    """
    gen_cols = []
    gen_scores = []
    
    for i, col in enumerate(columns):
        if s_init[i] == 0.0:
            gen_cols.append(col)
            gen_scores.append(y_pred[i])
            
    # 升序排列用于水平条形图的展示
    sorted_idx = np.argsort(gen_scores)
    sorted_cols = [gen_cols[idx] for idx in sorted_idx]
    sorted_scores = [gen_scores[idx] for idx in sorted_idx]
    
    plt.figure(figsize=(10, 12))
    
    # 建立基于风险分数的渐变彩色条
    norm_scores = (np.array(sorted_scores) - min(sorted_scores)) / (max(sorted_scores) - min(sorted_scores) + 1e-9)
    colors = plt.cm.plasma(norm_scores)
    
    bars = plt.barh(sorted_cols, sorted_scores, color=colors, edgecolor="black", height=0.6, alpha=0.9)
    
    # 标注条形图的具体预测分值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2.0, f"{width:.4f}", 
                 ha='left', va='center', fontsize=8, fontweight='semibold')
        
    plt.title("普通字段 (General) 经 GNN 传播学到的隐性风险排序", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("隐性敏感风险得分 (y_hat)", fontsize=11)
    plt.grid(axis='x', linestyle=":", alpha=0.6)
    plt.xlim(0, max(sorted_scores) + 0.06)
    
    plot_path = os.path.join(save_dir, "risk_rankings.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 已将风险排名条形图保存至: {plot_path}")

def plot_risk_propagation_graph(columns, y_pred, s_init, A_tilde, save_dir, top_edges_k=80):
    """
    使用 NetworkX 绘制风险传播图结构。
    """
    try:
        import networkx as nx
    except ImportError:
        print(" [警告] 未检测到 NetworkX 库，跳过拓扑关系图绘制。")
        return
        
    N = len(columns)
    G = nx.DiGraph()
    
    # 节点注册
    for i, col in enumerate(columns):
        is_conf = (s_init[i] == 1.0)
        G.add_node(i, name=col, is_conf=is_conf, risk=float(y_pred[i]))
        
    # 保留最强的 top_edges_k 条传播路径，避免构图过度拥挤
    edge_list = []
    for i in range(N):
        for j in range(N):
            if i != j and A_tilde[i, j] > 0.0:
                edge_list.append((j, i, float(A_tilde[i, j])))  # 风险由 j 传播至 i
                
    edge_list.sort(key=lambda x: x[2], reverse=True)
    top_edges = edge_list[:top_edges_k]
    
    for u, v, w in top_edges:
        G.add_edge(u, v, weight=w)
        
    plt.figure(figsize=(14, 14))
    # 采用圆环布局，呈现高质感、规整的电网数据特征视图
    pos = nx.circular_layout(G)
    
    conf_nodes = [n for n, d in G.nodes(data=True) if d['is_conf']]
    gen_nodes = [n for n, d in G.nodes(data=True) if not d['is_conf']]
    
    # 绘制敏感的机密节点（红色菱形）
    nx.draw_networkx_nodes(G, pos, nodelist=conf_nodes, node_shape='d', 
                           node_color='#d50000', node_size=600, 
                           edgecolors='black', linewidths=2.0, label='Confidential 核心敏感字段')
    
    # 绘制普通节点（渐变圆形）
    gen_risks = [G.nodes[n]['risk'] for n in gen_nodes]
    if len(gen_risks) > 0:
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=gen_nodes, node_shape='o',
                                       node_color=gen_risks, cmap=plt.cm.plasma,
                                       node_size=400, edgecolors='black', linewidths=1.0, 
                                       vmin=0.0, vmax=1.0, label='General 字段 (颜色梯度对应隐性风险)')
        
        cbar = plt.colorbar(nodes, shrink=0.5, pad=0.05)
        cbar.set_label('GNN预测的风险得分 (y_hat)', rotation=270, labelpad=15, fontsize=11)
        
    # 依据边传播权重动态绘制连接弧线的粗细和透明度
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    max_weight = max(weights) if len(weights) > 0 else 1.0
    
    for u, v, d in edges:
        w_norm = d['weight'] / max_weight
        width = 0.5 + w_norm * 4.0
        alpha = 0.15 + w_norm * 0.75
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                               alpha=alpha, edge_color='#607d8b', arrows=True, arrowsize=12)
        
    # 绘制标签名称
    labels = {n: d['name'] for n, d in G.nodes(data=True)}
    for n in labels:
        if len(labels[n]) > 25:
            labels[n] = labels[n][:22] + "..."
            
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif', font_weight='semibold')
    
    plt.title("GNN 学到的风险传播拓扑网络图\n(图中仅展示强相关联的传播通路)", fontsize=14, fontweight="bold", pad=20)
    plt.legend(loc='upper right', scatterpoints=1)
    plt.axis('off')
    
    plot_path = os.path.join(save_dir, "risk_propagation_network.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 已将风险传播拓扑图保存至: {plot_path}")

def generate_all_visualizations(data_dict, train_results, save_dir="outputs"):
    """
    自动创建输出目录并触发生成四张核心图表。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plot_training_history(train_results["history"], save_dir)
    plot_correlation_weights(train_results["alpha"], save_dir)
    plot_risk_rankings(data_dict["columns"], train_results["y_pred"], data_dict["s_init"], save_dir)
    plot_risk_propagation_graph(data_dict["columns"], train_results["y_pred"], data_dict["s_init"], 
                                train_results["A_tilde"], save_dir)
    print(f" 所有科学图表都已成功绘制并保存至目录: '{save_dir}/'")
