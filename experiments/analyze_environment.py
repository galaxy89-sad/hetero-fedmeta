# hetero-fedmeta/experiments/analyze_environment.py (最终通用版)

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy.stats import entropy, pearsonr

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# 【新】从torchvision导入两个数据集类
from torchvision.datasets import CIFAR10, MNIST


def analyze_stability_vs_data_uniqueness(config):
    """
    分析指定环境下，客户端网络稳定性与其本地数据分布独特性（熵）之间的关系。
    """
    print("--- 开始分析客户端稳定性 vs 数据独特性 ---")

    # --- 1. 根据配置，动态加载指定的环境文件 ---
    dataset_name = config['fl']['dataset']
    alpha = config['fl']['non_iid_alpha']
    num_clients=config['meta']['client_num']
    env_name = f"env_{dataset_name}_alpha{alpha}_clients{num_clients}"
    env_dir = project_root / "data" / env_name

    if not env_dir.exists():
        print(f"错误: 找不到环境目录 {env_dir}。")
        print(f"请先在 configs/config.yaml 中设置好 dataset='{dataset_name}' 和 non_iid_alpha={alpha},")
        print("然后运行 experiments/create_environment.py 来创建它。")
        return

    profiles_path = env_dir / "device_profiles.csv"
    partition_path = env_dir / "partition_map.json"

    print(f"加载设备画像: {profiles_path}")
    profile_df = pd.read_csv(profiles_path, index_col='client_id')

    print(f"加载数据划分: {partition_path}")
    with open(partition_path, 'r') as f:
        partition_map = json.load(f)

    # --- 2. 为每个客户端计算数据分布的熵 ---
    num_classes = config['fl']['num_classes']
    client_metrics = []

    print("正在为每个客户端计算数据熵...")

    # 【核心修改】根据配置加载正确的数据集标签
    print(f"为数据集 {dataset_name.upper()} 加载标签...")
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='./data', train=True, download=True)
        labels = np.array(dataset.targets)
    elif dataset_name == 'mnist':
        dataset = MNIST(root='./data', train=True, download=True)
        labels = np.array(dataset.targets)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    for client_id_str, indices in partition_map.items():
        client_id = int(client_id_str)

        # 获取该客户端的稳定性
        stability = profile_df.loc[client_id, 'stability']

        # 计算该客户端本地数据的标签分布
        client_labels = labels[indices]

        if len(client_labels) > 0:
            label_counts = np.bincount(client_labels, minlength=num_classes)
            label_distribution = label_counts / len(client_labels)

            # 计算归一化熵
            max_entropy = entropy([1 / num_classes] * num_classes)
            data_entropy = entropy(label_distribution + 1e-9)
            normalized_entropy = data_entropy / max_entropy

            client_metrics.append({
                'client_id': client_id,
                'stability': stability,
                'normalized_entropy': normalized_entropy
            })

    metrics_df = pd.DataFrame(client_metrics)

    # --- 3. 可视化数据熵的分布，以验证异构性 ---
    plt.figure(figsize=(15, 6))

    # 子图1: 熵的直方图
    plt.subplot(1, 2, 1)
    sns.histplot(data=metrics_df, x='normalized_entropy', kde=True, bins=20)
    plt.title(f'Distribution of Data Entropy (alpha={alpha})')
    plt.xlabel('Normalized Entropy (Higher is more IID)')
    plt.ylabel('Number of Clients')
    plt.grid(True)

    # 子图2: 稳定性 vs 熵的散点图
    plt.subplot(1, 2, 2)
    correlation, p_value = pearsonr(metrics_df['stability'], metrics_df['normalized_entropy'])
    sns.regplot(x='stability', y='normalized_entropy', data=metrics_df,
                scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': 'red'})
    plt.title(f'Stability vs. Entropy (Corr: {correlation:.2f}, p-val: {p_value:.2f})')
    plt.xlabel('Network Stability')
    plt.ylabel('Normalized Entropy')
    plt.grid(True)

    plt.tight_layout()

    # 创建结果保存目录
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"analysis_{env_name}.png"

    plt.savefig(output_path)
    print(f"\n分析图已保存到: {output_path}")
    plt.show()


if __name__ == '__main__':
    # 加载主配置文件，根据其设置来分析对应的环境
    with open(project_root / "configs/config.yaml", 'r') as f:
        main_config = yaml.safe_load(f)

    analyze_stability_vs_data_uniqueness(main_config)