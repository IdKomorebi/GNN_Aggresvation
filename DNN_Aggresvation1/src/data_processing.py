import os
import json
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import normalized_mutual_info_score

# 1. 核心字段定义（符合 LaTeX 规范设计）
D_DROP = [
    "datetime_beginning_utc",
    "datetime_beginning_ept",
    "net_sched_interchange_mw",
    "prelim_load_avg_hourly",
    "total_pjm_rt_load_mwh",
    "wind_generation_mw",
    "solar_generation_mw",
    "da_as_as_mw_primary_reserve",
    "da_as_as_mw_synchronized_reserve",
    "da_as_as_mw_thirty_minutes_reserve",
    "system_energy_price_rt"
]

C_CONFIDENTIAL = [
    "net_actual_interchange_mw",
    "gross_actual_interchange_mw",
    "total_gen",
    "metered_load_mw",
    "total_losses",
    "congestion_price_da",
    "congestion_price_rt",
    "marginal_loss_price_da",
    "total_lmp_da",
    "da_as_total_mw_primary_reserve",
    "da_as_total_mw_synchronized_reserve",
    "da_as_total_mw_thirty_minutes_reserve"
]

def load_and_preprocess_data(csv_path):
    """
    加载并清理 CSV 数据集，删除冗余列并返回处理后的 DataFrame。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据集文件: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 丢弃指定的冗余字段
    cols_to_drop = [c for c in D_DROP if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    # 转换为数值类型
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    return df_clean

def extract_node_features(df):
    """
    提取每个字段的 13 维统计特征 (z_i)，并在节点间进行列标准化。
    返回标准化特征矩阵、敏感位标签和融合了敏感度的 h_raw 特征 [N, 14]。
    """
    N = df.shape[1]
    features = []
    
    for col_name in df.columns:
        x = df[col_name].values
        
        # 1. 均值 (mu)
        mu = np.nanmean(x) if not np.isnan(x).all() else 0.0
        # 2. 标准差 (sigma)
        sigma = np.nanstd(x) if not np.isnan(x).all() else 0.0
        # 3. 最大值
        x_max = np.nanmax(x) if not np.isnan(x).all() else 0.0
        # 4. 最小值
        x_min = np.nanmin(x) if not np.isnan(x).all() else 0.0
        # 5. 中位数
        median = np.nanmedian(x) if not np.isnan(x).all() else 0.0
        # 6. 25%分位数 (q25)
        q25 = np.nanpercentile(x, 25) if not np.isnan(x).all() else 0.0
        # 7. 75%分位数 (q75)
        q75 = np.nanpercentile(x, 75) if not np.isnan(x).all() else 0.0
        # 8. 缺失率
        miss = np.isnan(x).mean()
        
        # 一阶差分
        diff_x = np.diff(x)
        # 9. 差分均值
        diffmean = np.nanmean(diff_x) if not np.isnan(diff_x).all() else 0.0
        # 10. 差分标准差
        diffstd = np.nanstd(diff_x) if not np.isnan(diff_x).all() else 0.0
        
        # 11. 滞后一阶自相关系数
        series = pd.Series(x).dropna()
        if len(series) > 1:
            autocorr1 = series.autocorr(lag=1)
            if np.isnan(autocorr1):
                autocorr1 = 0.0
        else:
            autocorr1 = 0.0
            
        # 12. 偏度 (skew)
        skew = scipy.stats.skew(x, nan_policy='omit')
        if np.isnan(skew):
            skew = 0.0
            
        # 13. 峰度 (kurt)
        kurt = scipy.stats.kurtosis(x, nan_policy='omit')
        if np.isnan(kurt):
            kurt = 0.0
            
        features.append([mu, sigma, x_max, x_min, median, q25, q75, miss, diffmean, diffstd, autocorr1, skew, kurt])
        
    z = np.array(features, dtype=np.float32)  # 形状: [N, 13]
    
    # 沿列方向跨节点标准化 (按列标准化)
    z_mean = np.nanmean(z, axis=0, keepdims=True)
    z_std = np.nanstd(z, axis=0, keepdims=True)
    z_std[z_std == 0] = 1.0  # 避免除以零
    z_tilde = (z - z_mean) / z_std
    
    # 敏感性初值 (Confidential = 1.0, General = 0.0)
    s_init = np.array([1.0 if c in C_CONFIDENTIAL else 0.0 for c in df.columns], dtype=np.float32)
    
    # 拼接敏感度和标准化特征矩阵得到 h_raw，形状为 [N, 14]
    h_raw = np.column_stack([s_init, z_tilde])
    
    return z_tilde, s_init, h_raw

def compute_distance_correlation_matrix(df_downsampled):
    """
    基于我们设计的矩阵向量化双重中心化公式，在毫秒级内极速计算所有字段的距离相关系数 (dCor) 矩阵。
    形状: [N, N], 值范围: [0, 1]
    """
    N = df_downsampled.shape[1]
    n = df_downsampled.shape[0]
    
    M = np.zeros((N, n * n), dtype=np.float32)
    
    for i in range(N):
        x = df_downsampled.iloc[:, i].values
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=np.nanmean(x) if not np.isnan(x).all() else 0.0)
            
        # 欧式距离矩阵
        dist_matrix = np.abs(x[:, None] - x[None, :])
        
        # 双重中心化 (Double Centering)
        row_means = dist_matrix.mean(axis=1, keepdims=True)
        col_means = dist_matrix.mean(axis=0, keepdims=True)
        grand_mean = dist_matrix.mean()
        
        A = dist_matrix - row_means - col_means + grand_mean
        M[i, :] = A.flatten()
        
    # 一步计算距离协方差矩阵 (dCov^2)
    dcov2 = (M @ M.T) / (n * n)
    
    # 对角线元素为距离方差 (dVar^2)
    dvar2 = np.diag(dcov2)
    
    # 距离相关系数计算 (dCor)
    dcov = np.sqrt(np.clip(dcov2, 0, None))
    dvar = np.sqrt(np.clip(dvar2, 0, None))
    
    denom = np.sqrt(dvar[:, None] @ dvar[None, :] + 1e-9)
    dcor = dcov / denom
    np.fill_diagonal(dcor, 1.0)
    return np.clip(dcor, 0.0, 1.0)

def compute_multi_correlation(df, downsample_size=300):
    """
    并行/快速计算 5 种相关性测度矩阵 (Pearson, Spearman, Kendall, NMI, dCor)。
    对于 Kendall、NMI 和 dCor 等慢速指标，自动进行采样加速。
    返回的矩阵 R 形状为 [5, N, N]。
    """
    N = df.shape[1]
    
    # 线性插值填充空值
    df_filled = df.interpolate(method='linear', limit_direction='both').fillna(0.0)
    
    # 1. Pearson 相关系数
    pearson = np.abs(df_filled.corr(method='pearson').values)
    pearson = np.nan_to_num(pearson, nan=0.0)
    np.fill_diagonal(pearson, 1.0)
    
    # 2. Spearman 相关系数
    spearman = np.abs(df_filled.corr(method='spearman').values)
    spearman = np.nan_to_num(spearman, nan=0.0)
    np.fill_diagonal(spearman, 1.0)
    
    # 降采样做慢速复杂指标计算
    if len(df_filled) > downsample_size:
        indices = np.linspace(0, len(df_filled)-1, downsample_size, dtype=int)
        df_down = df_filled.iloc[indices]
    else:
        df_down = df_filled
        
    # 3. Kendall's Tau 相关系数
    kendall = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        kendall[i, i] = 1.0
        for j in range(i + 1, N):
            tau, _ = scipy.stats.kendalltau(df_down.iloc[:, i], df_down.iloc[:, j])
            val = np.abs(tau) if not np.isnan(tau) else 0.0
            kendall[i, j] = val
            kendall[j, i] = val
            
    # 4. 归一化互信息 (NMI)
    nmi = np.zeros((N, N), dtype=np.float32)
    df_binned = pd.DataFrame()
    for col in df_down.columns:
        try:
            bins = pd.cut(df_down[col], bins=10, labels=False, duplicates='drop')
            df_binned[col] = bins.fillna(-1)
        except Exception:
            df_binned[col] = 0
            
    for i in range(N):
        nmi[i, i] = 1.0
        for j in range(i + 1, N):
            score = normalized_mutual_info_score(df_binned.iloc[:, i], df_binned.iloc[:, j])
            val = float(score) if not np.isnan(score) else 0.0
            nmi[i, j] = val
            nmi[j, i] = val
            
    # 5. 距离相关系数 (dCor)
    dcor = compute_distance_correlation_matrix(df_down)
    
    # 拼装成 [5, N, N] 形状的大矩阵
    R = np.stack([pearson, spearman, kendall, nmi, dcor], axis=0).astype(np.float32)
    return R

def construct_graph_structure(R, K_neighbors=5, theta=0.5):
    """
    计算相关性指标均值 r_bar，并生成固定的 Top-K 与阈值融合的候选传播边掩码 m_ij。
    """
    r_bar = np.mean(R, axis=0)  # 形状: [N, N]
    N = r_bar.shape[0]
    
    m = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        # 寻找节点 i 的 Top-K 强关联邻居（剔除自身）
        r_row = r_bar[i].copy()
        r_row[i] = -1.0  
        top_k_indices = np.argsort(r_row)[-K_neighbors:]
        
        for j in range(N):
            if i == j:
                m[i, j] = 0.0
            elif j in top_k_indices or r_bar[i, j] >= theta:
                m[i, j] = 1.0
                
    return r_bar, m

def _max_topk_blend(values, top_k=3, max_weight=0.7):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0

    max_val = float(np.max(values))
    top_vals = np.sort(values)[::-1][:top_k]
    top_mean = float(np.mean(top_vals)) if top_vals.size > 0 else 0.0
    return max_weight * max_val + (1.0 - max_weight) * top_mean

def _predictive_r2_dnn_univariate(x, y, test_ratio=0.3, epochs=80,
                                  hidden_dim=16, lr=0.01,
                                  weight_decay=1e-4, seed=2026,
                                  epsilon=1e-12):
    """
    用单个 General 字段 x 训练一个小型 DNN 去预测单个 Confidential 字段 y，
    并在时间后段测试集上计算 R^2。
    q_{i,c}=max(0, R^2)，负 R^2 表示不如均值基线，按 0 处理。
    """
    n = len(x)
    if n < 10:
        return 0.0

    test_ratio = float(np.clip(test_ratio, 0.05, 0.5))
    split_idx = int(round(n * (1.0 - test_ratio)))
    split_idx = min(max(split_idx, 2), n - 2)

    x_train = np.asarray(x[:split_idx], dtype=np.float64)
    x_test = np.asarray(x[split_idx:], dtype=np.float64)
    y_train = np.asarray(y[:split_idx], dtype=np.float64)
    y_test = np.asarray(y[split_idx:], dtype=np.float64)

    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if x_std <= epsilon or y_std <= epsilon:
        return 0.0

    x_train_z = ((x_train - x_mean) / x_std).astype(np.float32).reshape(-1, 1)
    x_test_z = ((x_test - x_mean) / x_std).astype(np.float32).reshape(-1, 1)
    y_train_z = ((y_train - y_mean) / y_std).astype(np.float32).reshape(-1, 1)

    torch.manual_seed(int(seed))
    model = nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    x_train_tensor = torch.tensor(x_train_z, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_z, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_z, dtype=torch.float32)

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_z = model(x_test_tensor).squeeze(-1).cpu().numpy()

    y_pred = y_pred_z * y_std + y_mean
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    if ss_tot <= epsilon:
        return 0.0

    ss_res = np.sum((y_test - y_pred) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + epsilon)
    if not np.isfinite(r2):
        return 0.0
    return float(max(0.0, r2))

def compute_general_predictive_scores(df, gen_indices, conf_indices,
                                      test_ratio=0.3, dnn_epochs=80,
                                      dnn_hidden_dim=16, dnn_lr=0.01,
                                      dnn_weight_decay=1e-4, dnn_seed=2026):
    """
    对每个 General 字段 x_i，分别预测每个 Confidential 字段 x_c。
    每个 (x_i, x_c) 对应一个 DNN 回归器，先得到 q_{i,c}=max(0, R^2_DNN(x_i -> x_c))，再聚合：
    q_i = 0.7 * max_c q_{i,c} + 0.3 * MeanTop3_c(q_{i,c})。
    """
    df_filled = df.interpolate(method='linear', limit_direction='both').fillna(0.0)
    values = df_filled.values.astype(np.float64)
    N = df.shape[1]
    q_all = np.zeros(N, dtype=np.float32)
    q_matrix = np.zeros((N, N), dtype=np.float32)

    for idx in gen_indices:
        q_ic_list = []
        x = values[:, idx]
        for c_idx in conf_indices:
            pair_seed = int(dnn_seed) + int(idx) * 1009 + int(c_idx)
            q_ic = _predictive_r2_dnn_univariate(
                x,
                values[:, c_idx],
                test_ratio=test_ratio,
                epochs=dnn_epochs,
                hidden_dim=dnn_hidden_dim,
                lr=dnn_lr,
                weight_decay=dnn_weight_decay,
                seed=pair_seed
            )
            q_matrix[idx, c_idx] = q_ic
            q_ic_list.append(q_ic)
        q_all[idx] = _max_topk_blend(q_ic_list)

    for idx in conf_indices:
        q_all[idx] = 1.0
        q_matrix[idx, idx] = 1.0

    return q_all, q_matrix

def _dnn_q_cache_signature(columns, conf_indices, test_ratio, dnn_epochs,
                           dnn_hidden_dim, dnn_lr, dnn_weight_decay, dnn_seed):
    return {
        "columns": list(columns),
        "confidential_columns": [columns[i] for i in conf_indices],
        "predictive_test_ratio": float(test_ratio),
        "dnn_epochs": int(dnn_epochs),
        "dnn_hidden_dim": int(dnn_hidden_dim),
        "dnn_lr": float(dnn_lr),
        "dnn_weight_decay": float(dnn_weight_decay),
        "dnn_seed": int(dnn_seed)
    }

def _load_dnn_q_cache(cache_dir, columns, conf_indices, test_ratio, dnn_epochs,
                      dnn_hidden_dim, dnn_lr, dnn_weight_decay, dnn_seed):
    if cache_dir is None:
        return None

    metadata_path = os.path.join(cache_dir, "metadata.json")
    scores_path = os.path.join(cache_dir, "dnn_q_scores.csv")
    matrix_path = os.path.join(cache_dir, "general_to_confidential_predictive_r2.csv")
    if not (os.path.exists(metadata_path) and os.path.exists(scores_path) and os.path.exists(matrix_path)):
        return None

    expected = _dnn_q_cache_signature(
        columns, conf_indices, test_ratio, dnn_epochs, dnn_hidden_dim,
        dnn_lr, dnn_weight_decay, dnn_seed
    )
    with open(metadata_path, "r", encoding="utf-8") as f:
        cached = json.load(f)
    if cached != expected:
        return None

    score_df = pd.read_csv(scores_path)
    matrix_df = pd.read_csv(matrix_path)
    q_all = np.zeros(len(columns), dtype=np.float32)
    q_matrix = np.zeros((len(columns), len(columns)), dtype=np.float32)

    column_to_idx = {col: idx for idx, col in enumerate(columns)}
    for _, row in score_df.iterrows():
        idx = column_to_idx.get(row["字段名称"])
        if idx is not None:
            q_all[idx] = float(row["聚合推断能力q_i"])

    for _, row in matrix_df.iterrows():
        idx = column_to_idx.get(row["General字段"])
        if idx is None:
            continue
        for c_idx in conf_indices:
            key = f"q_to_{columns[c_idx]}"
            if key in row:
                q_matrix[idx, c_idx] = float(row[key])

    for idx in conf_indices:
        q_all[idx] = 1.0
        q_matrix[idx, idx] = 1.0

    return q_all, q_matrix

def _save_dnn_q_cache(cache_dir, columns, gen_indices, conf_indices, q_all, q_matrix,
                      test_ratio, dnn_epochs, dnn_hidden_dim, dnn_lr,
                      dnn_weight_decay, dnn_seed):
    if cache_dir is None:
        return

    os.makedirs(cache_dir, exist_ok=True)
    metadata = _dnn_q_cache_signature(
        columns, conf_indices, test_ratio, dnn_epochs, dnn_hidden_dim,
        dnn_lr, dnn_weight_decay, dnn_seed
    )
    with open(os.path.join(cache_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    score_rows = []
    for idx in gen_indices:
        score_rows.append({
            "字段名称": columns[idx],
            "聚合推断能力q_i": float(q_all[idx])
        })
    pd.DataFrame(score_rows).to_csv(os.path.join(cache_dir, "dnn_q_scores.csv"), index=False)

    matrix_rows = []
    for idx in gen_indices:
        row = {
            "General字段": columns[idx],
            "聚合推断能力q_i": float(q_all[idx])
        }
        for c_idx in conf_indices:
            row[f"q_to_{columns[c_idx]}"] = float(q_matrix[idx, c_idx])
        matrix_rows.append(row)
    pd.DataFrame(matrix_rows).to_csv(
        os.path.join(cache_dir, "general_to_confidential_predictive_r2.csv"),
        index=False
    )

    with open(os.path.join(cache_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# DNN-q 缓存\n\n")
        f.write("该目录缓存 General 字段到 Confidential 字段的 DNN 样本外预测 R²。\n\n")
        f.write("- `dnn_q_scores.csv`: 每个 General 字段聚合后的 `q_i`。\n")
        f.write("- `general_to_confidential_predictive_r2.csv`: 每个 General 到每个 Confidential 的 `q_{i,c}`。\n")
        f.write("- `metadata.json`: 生成该缓存时使用的字段和 DNN 参数签名。\n")

def construct_supervision_labels(df, R, r_bar, m, general_label_count=10,
                                 high_risk_ratio=0.7, inference_weight=0.65,
                                 predictive_test_ratio=0.3, dnn_epochs=80,
                                 dnn_hidden_dim=16, dnn_lr=0.01,
                                 dnn_weight_decay=1e-4, dnn_seed=2026,
                                 dnn_q_cache_dir=None):
    """
    构建半监督标签：
    - Confidential 字段全部作为强监督标签，y_i = 1.0。
    - General 字段仅选取少量高/低风险锚点参与监督。
    - General 伪标签由推断能力分数与初始敏感度传播分数融合得到。
    """
    N = df.shape[1]
    columns = df.columns
    
    conf_indices = [i for i, c in enumerate(columns) if c in C_CONFIDENTIAL]
    gen_indices = [i for i, c in enumerate(columns) if c not in C_CONFIDENTIAL]
    
    y = np.zeros(N, dtype=np.float32)
    inference_score = np.zeros(N, dtype=np.float32)
    sensitivity_score = np.zeros(N, dtype=np.float32)
    supervision_mask = np.zeros(N, dtype=np.float32)
    label_source = np.array(["unlabeled_general"] * N, dtype=object)
    
    for idx in conf_indices:
        y[idx] = 1.0
        inference_score[idx] = 1.0
        sensitivity_score[idx] = 1.0
        supervision_mask[idx] = 1.0
        label_source[idx] = "confidential_seed"

    q_matrix = np.zeros((N, N), dtype=np.float32)

    if len(conf_indices) == 0:
        pseudo_y = y.copy()
        return y, inference_score, pseudo_y, inference_score, sensitivity_score, supervision_mask, label_source, q_matrix

    cache_result = _load_dnn_q_cache(
        dnn_q_cache_dir,
        list(columns),
        conf_indices,
        predictive_test_ratio,
        dnn_epochs,
        dnn_hidden_dim,
        dnn_lr,
        dnn_weight_decay,
        dnn_seed
    )
    if cache_result is None:
        inference_score, q_matrix = compute_general_predictive_scores(
            df,
            gen_indices,
            conf_indices,
            test_ratio=predictive_test_ratio,
            dnn_epochs=dnn_epochs,
            dnn_hidden_dim=dnn_hidden_dim,
            dnn_lr=dnn_lr,
            dnn_weight_decay=dnn_weight_decay,
            dnn_seed=dnn_seed
        )
        _save_dnn_q_cache(
            dnn_q_cache_dir,
            list(columns),
            gen_indices,
            conf_indices,
            inference_score,
            q_matrix,
            predictive_test_ratio,
            dnn_epochs,
            dnn_hidden_dim,
            dnn_lr,
            dnn_weight_decay,
            dnn_seed
        )
    else:
        inference_score, q_matrix = cache_result

    A_seed = m * r_bar
    row_sums = np.sum(A_seed, axis=1, keepdims=True)
    A_seed_tilde = A_seed / (row_sums + 1e-8)
    s_seed = np.zeros(N, dtype=np.float32)
    s_seed[conf_indices] = 1.0
    sensitivity_score = np.clip(A_seed_tilde @ s_seed, 0.0, 1.0).astype(np.float32)
    sensitivity_score[conf_indices] = 1.0

    inference_weight = float(np.clip(inference_weight, 0.0, 1.0))
    pseudo_y = inference_weight * inference_score + (1.0 - inference_weight) * sensitivity_score
    pseudo_y = np.clip(pseudo_y, 0.0, 1.0).astype(np.float32)
    pseudo_y[conf_indices] = 1.0
    y[:] = pseudo_y

    if len(gen_indices) > 0 and general_label_count > 0:
        label_count = min(int(general_label_count), len(gen_indices))
        high_count = int(np.ceil(label_count * float(np.clip(high_risk_ratio, 0.0, 1.0))))
        low_count = label_count - high_count

        gen_indices_arr = np.array(gen_indices, dtype=int)
        sorted_gen = gen_indices_arr[np.argsort(pseudo_y[gen_indices_arr])]
        selected_low = sorted_gen[:low_count] if low_count > 0 else np.array([], dtype=int)
        selected_high = sorted_gen[-high_count:] if high_count > 0 else np.array([], dtype=int)
        selected = np.unique(np.concatenate([selected_low, selected_high]))

        if selected.size < label_count:
            remaining = [idx for idx in sorted_gen[::-1] if idx not in set(selected.tolist())]
            fill = np.array(remaining[:label_count - selected.size], dtype=int)
            selected = np.unique(np.concatenate([selected, fill]))

        supervision_mask[selected] = 1.0
        for idx in selected_low:
            label_source[idx] = "general_pseudo_low"
        for idx in selected_high:
            label_source[idx] = "general_pseudo_high"

    return y, inference_score, pseudo_y, inference_score, sensitivity_score, supervision_mask, label_source, q_matrix

def prepare_pipeline_data(csv_path, K_neighbors=5, theta=0.5, downsample_size=300,
                          general_label_count=10, high_risk_ratio=0.7,
                          inference_weight=0.65, predictive_test_ratio=0.3,
                          dnn_epochs=80, dnn_hidden_dim=16, dnn_lr=0.01,
                          dnn_weight_decay=1e-4, dnn_seed=2026,
                          dnn_q_cache_dir=None):
    """
    统一驱动整个数据流管道，返回包含预处理全量数据的字典。
    """
    df_clean = load_and_preprocess_data(csv_path)
    z_tilde, s_init, h_raw = extract_node_features(df_clean)
    R = compute_multi_correlation(df_clean, downsample_size=downsample_size)
    r_bar, m = construct_graph_structure(R, K_neighbors=K_neighbors, theta=theta)
    y, q_all, pseudo_y, inference_score, sensitivity_score, supervision_mask, label_source, q_matrix = construct_supervision_labels(
        df_clean,
        R,
        r_bar,
        m,
        general_label_count=general_label_count,
        high_risk_ratio=high_risk_ratio,
        inference_weight=inference_weight,
        predictive_test_ratio=predictive_test_ratio,
        dnn_epochs=dnn_epochs,
        dnn_hidden_dim=dnn_hidden_dim,
        dnn_lr=dnn_lr,
        dnn_weight_decay=dnn_weight_decay,
        dnn_seed=dnn_seed,
        dnn_q_cache_dir=dnn_q_cache_dir
    )
    
    columns = list(df_clean.columns)
    
    return {
        "df": df_clean,
        "columns": columns,
        "h_raw": h_raw,
        "s_init": s_init,
        "z_tilde": z_tilde,
        "R": R,
        "r_bar": r_bar,
        "m": m,
        "y": y,
        "q_all": q_all,
        "pseudo_y": pseudo_y,
        "inference_score": inference_score,
        "sensitivity_score": sensitivity_score,
        "supervision_mask": supervision_mask,
        "label_source": label_source,
        "predictive_r2_matrix": q_matrix
    }
