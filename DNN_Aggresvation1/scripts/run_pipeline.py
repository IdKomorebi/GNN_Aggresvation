import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 动态配置项目根目录 (DNN_Aggresvation1)，确保能够从任何位置执行本脚本
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. 引入重构后的模块包 (src)
from src.data_processing import prepare_pipeline_data
from src.train import train_gnn_model
from src.visualize import generate_all_visualizations

def load_yaml_config(config_path):
    """
    加载 YAML 配置文件。
    优先采用 PyYAML 库解析；如果环境未配置 PyYAML，则触发我们编写的零依赖 YAML 分析器作为容错备份。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except ImportError:
        # Fallback 容错备份：手动轻量级 YAML 解析器
        config = {}
        current_section = None
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 剔除首尾空格和注释
                line_clean = line.strip()
                if not line_clean or line_clean.startswith('#'):
                    continue
                    
                # 识别小节标题 (如 data:)
                if line_clean.endswith(':'):
                    current_section = line_clean[:-1].strip()
                    config[current_section] = {}
                    continue
                    
                if ':' in line_clean:
                    parts = line_clean.split(':', 1)
                    k = parts[0].strip()
                    # 剥离行末可能存在的尾部注释
                    v = parts[1].split('#')[0].strip()
                    
                    # 清理引号
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    else:
                        # 转换基本数据类型
                        try:
                            if '.' in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            pass
                            
                    if current_section:
                        config[current_section][k] = v
                    else:
                        config[k] = v
        return config

def resolve_path(relative_path):
    """
    将路径归一化为相对于项目根目录 (DNN_Aggresvation1) 的绝对路径，
    保证无论从任何目录启动脚本，文件读取和写入路径都能 100% 寻址成功。
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, relative_path))

def save_run_config(config, config_path, output_dir):
    """
    将本次运行使用的关键设置保存为 Markdown 说明。
    """
    config_out_path = os.path.join(output_dir, "run_config.md")
    with open(config_out_path, "w", encoding="utf-8") as f:
        f.write("# 本次运行说明\n\n")
        f.write(f"- 源配置文件：`{config_path}`\n")
        f.write(f"- 运行输出目录：`{output_dir}`\n\n")
        for section, values in config.items():
            f.write(f"## {section}\n\n")
            if isinstance(values, dict):
                for key, value in values.items():
                    f.write(f"- `{key}`: `{value}`\n")
            else:
                f.write(f"- `{values}`\n")
            f.write("\n")
    return config_out_path

def main():
    parser = argparse.ArgumentParser(description="模块化 GNN 风险聚合运行脚本 - DNN_Aggresvation1")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"),
                        help="YAML配置文件的路径")
    args = parser.parse_args()
    
    # 3. 读取配置文件
    config_path = os.path.abspath(args.config)
    print(f"正在加载配置文件: {config_path}")
    config = load_yaml_config(config_path)
    
    # 4. 路径归一化解析
    csv_path = resolve_path(config["data"]["csv_path"])
    output_base_dir = resolve_path(config["outputs"]["dir"])
    run_id = datetime.now().strftime("run%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    run_config_path = save_run_config(config, config_path, output_dir)
    
    # 提取训练参数
    downsample_size = config["data"]["downsample_size"]
    pseudo_cfg = config.get("pseudo_label", {})
    general_label_count = pseudo_cfg.get("general_label_count", 10)
    high_risk_ratio = pseudo_cfg.get("high_risk_ratio", 0.7)
    inference_weight = pseudo_cfg.get("inference_weight", 0.65)
    predictive_test_ratio = pseudo_cfg.get("predictive_test_ratio", 0.3)
    dnn_epochs = pseudo_cfg.get("dnn_epochs", 80)
    dnn_hidden_dim = pseudo_cfg.get("dnn_hidden_dim", 16)
    dnn_lr = pseudo_cfg.get("dnn_lr", 0.01)
    dnn_weight_decay = pseudo_cfg.get("dnn_weight_decay", 1e-4)
    dnn_seed = pseudo_cfg.get("dnn_seed", 2026)
    dnn_q_cache_dir = pseudo_cfg.get("dnn_q_cache_dir", os.path.join(config["outputs"]["dir"], "dnn_q_cache"))
    dnn_q_cache_dir = resolve_path(dnn_q_cache_dir)
    k_neighbors = config["graph"]["k_neighbors"]
    theta = config["graph"]["theta"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    beta_lr = config["training"].get("beta_lr", None)
    alpha_temperature = config["training"].get("alpha_temperature", 1.0)
    w_c = config["training"]["w_c"]
    lmbda = config["training"]["lmbda"]
    seed = config["training"].get("seed", None)
    
    print("\n" + "="*70)
    print("      启动重构模块化之后的 GNN 风险传播与聚合流水线")
    print("="*70)
    print(f"项目根目录:        {PROJECT_ROOT}")
    print(f"数据源 CSV 路径:   {csv_path}")
    print(f"训练 Epochs 轮数:  {epochs}")
    print(f"General 监督锚点:  {general_label_count}")
    print(f"预测R²测试集比例:   {predictive_test_ratio}")
    print(f"DNN推断模型:       epochs={dnn_epochs}, hidden={dnn_hidden_dim}, lr={dnn_lr}")
    print(f"DNN-q 缓存目录:    {dnn_q_cache_dir}")
    print(f"beta 单独学习率:    {beta_lr}")
    print(f"alpha softmax温度: {alpha_temperature}")
    print(f"融合参数正则项:     {lmbda}")
    print(f"随机种子:          {seed}")
    print(f"结果输出存储目录:  {output_dir}")
    print(f"本次运行说明:      {run_config_path}")
    print("="*70 + "\n")
    
    # 5. 驱动数据准备
    print("[1/4] 启动数据读取、特征提取与 5 相关性矩阵构建...")
    data_dict = prepare_pipeline_data(
        csv_path=csv_path,
        K_neighbors=k_neighbors,
        theta=theta,
        downsample_size=downsample_size,
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
    print(" 预处理与建图矩阵构建顺利完成！")
    
    # 6. 模型训练与收敛
    print("\n[2/4] 载入 PyTorch 神经网络并开始收敛训练...")
    train_results = train_gnn_model(
        data_dict=data_dict,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        epochs=epochs,
        w_c=w_c,
        lmbda=lmbda,
        seed=seed,
        beta_lr=beta_lr,
        alpha_temperature=alpha_temperature,
        verbose=True
    )
    
    # 7. 可视化图表生成
    print("[3/4] 渲染多维度风险评估可视化图表...")
    generate_all_visualizations(data_dict, train_results, save_dir=output_dir)
    
    # 8. 导出数据评估报告与排名
    print("\n[4/4] 编译最终电网数据隐性风险排名 CSV 报告...")
    os.makedirs(output_dir, exist_ok=True)
    
    columns = data_dict["columns"]
    y_pred = train_results["y_pred"]
    s_init = data_dict["s_init"]
    supervision_mask = data_dict["supervision_mask"]
    label_source = data_dict["label_source"]
    pseudo_y = data_dict["pseudo_y"]
    inference_score = data_dict["inference_score"]
    sensitivity_score = data_dict["sensitivity_score"]
    
    results_list = []
    for i, col in enumerate(columns):
        status = "Confidential" if s_init[i] == 1.0 else "General"
        supervised = bool(supervision_mask[i] > 0.0)
        results_list.append({
            "字段名称": col,
            "安全分类": status,
            "监督状态": "Supervised" if supervised else "Unlabeled",
            "标签来源": str(label_source[i]),
            "监督标签_y": float(data_dict["y"][i]) if supervised else np.nan,
            "全量伪风险分数_pseudo_y": float(pseudo_y[i]),
            "预测R2推断能力_q": float(inference_score[i]),
            "初始敏感传播分数_seed_propagation": float(sensitivity_score[i]),
            "GNN预测隐性风险值_y_hat": float(y_pred[i])
        })
        
    df_results = pd.DataFrame(results_list)
    df_results_sorted = df_results.sort_values(by="GNN预测隐性风险值_y_hat", ascending=False)
    
    csv_out_path = os.path.join(output_dir, "risk_assessment_rankings.csv")
    df_results_sorted.to_csv(csv_out_path, index=False)
    print(f" 已将完整字段预测排名 CSV 导出至: '{csv_out_path}'")

    q_matrix = data_dict["predictive_r2_matrix"]
    conf_cols = [col for i, col in enumerate(columns) if s_init[i] == 1.0]
    gen_rows = []
    for i, col in enumerate(columns):
        if s_init[i] == 0.0:
            row = {"General字段": col, "聚合推断能力q_i": float(inference_score[i])}
            for c_idx, conf_col in enumerate(columns):
                if s_init[c_idx] == 1.0:
                    row[f"q_to_{conf_col}"] = float(q_matrix[i, c_idx])
            gen_rows.append(row)
    q_out_path = os.path.join(output_dir, "general_to_confidential_predictive_r2.csv")
    pd.DataFrame(gen_rows).to_csv(q_out_path, index=False)
    print(f" 已将 General->Confidential 预测R²矩阵导出至: '{q_out_path}'")
    
    # 控制台打印前 10 危险的普通字段
    print("\n" + "#"*60)
    print("      GNN 模型识别出的前 10 位高风险隐性泄漏字段 (General)")
    print("#"*60)
    
    df_gen = df_results_sorted[df_results_sorted["安全分类"] == "General"].head(10)
    
    print(f"{'排名':<4} | {'字段特征名称':<42} | {'风险评分 (y_hat)':<10}")
    print("-" * 65)
    for rank, (_, row) in enumerate(df_gen.iterrows(), 1):
        print(f"{rank:<4} | {row['字段名称']:<42} | {row['GNN预测隐性风险值_y_hat']:.6f}")
    print("#"*60 + "\n")
    
    alpha = train_results["alpha"]
    print("学到的多指标相关性融合系数 (alpha):")
    print(f"  - Pearson 相关系数:           {alpha[0]:.6f}")
    print(f"  - Spearman 秩相关系数:        {alpha[1]:.6f}")
    print(f"  - Kendall's Tau 秩相关系数:   {alpha[2]:.6f}")
    print(f"  - 归一化互信息 (NMI) 指数:    {alpha[3]:.6f}")
    print(f"  - 距离相关系数 (dCor) 指数:   {alpha[4]:.6f}")
    print("="*70)
    print(" 恭喜！重构版 GNN 风险传播流水线全部执行完毕！")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
