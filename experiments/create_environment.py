# hetero-fedmeta/experiments/create_environment.py (仅修复数量问题版)

import torch
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import json
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# 从我们已有的模块中导入所需函数和类
from utils.device_factory import load_generator
from utils.data_partition import partition_data_non_iid  # <-- 保持使用狄利克雷划分
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms


def create_and_save_environment(config):
    """
    创建并保存一个完整的、可复现的联邦学习环境。
    确保设备画像和数据划分的数量一致。
    """
    print("--- 开始创建并保存联邦学习环境 ---")

    # --- 1. 根据配置，动态构建环境路径 ---
    dataset_name = config['fl']['dataset']
    alpha = config['fl']['non_iid_alpha']
    num_clients = config['meta']['client_num']

    env_name = f"env_{dataset_name}_alpha{alpha}_clients{num_clients}"
    output_dir = project_root / "data" / env_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device_profiles_path = output_dir / "device_profiles.csv"
    partition_map_path = output_dir / "partition_map.json"
    print(f"本次创建的环境将保存在: {output_dir}")

    # --- 2. 【核心修复】每次都根据当前的 client_num 重新生成设备画像 ---
    print(f"\n步骤1: 生成 {num_clients} 个设备画像...")
    gan_config = config['gan']
    checkpoint_path = project_root / gan_config['paths']['checkpoint_dir'] / "generator_epoch_200.pth"
    generator = load_generator(checkpoint_path)

    # 生成与客户端数量匹配的噪声
    noise = torch.randn(num_clients, 100).to(next(generator.parameters()).device)
    with torch.no_grad():
        profiles = generator(noise).cpu().numpy()

    profile_df = pd.DataFrame(profiles, columns=['power', 'bandwidth', 'ram', 'stability', 'energy'])
    profile_df.index.name = 'client_id'
    profile_df.to_csv(device_profiles_path)
    print(f"设备画像已保存到: {device_profiles_path}")

    # --- 3. 划分并保存与客户端数量匹配的数据分配 ---
    print(f"\n步骤2: 划分 {dataset_name.upper()} 数据集...")
    fl_config = config['fl']

    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    # 使用我们一直以来的狄利克雷划分方法
    client_datasets_subsets = partition_data_non_iid(
        train_dataset,
        num_clients=num_clients,
        alpha=fl_config['non_iid_alpha']
    )

    partition_map = {
        int(client_id): [int(i) for i in subset.indices]
        for client_id, subset in enumerate(client_datasets_subsets)
    }

    with open(partition_map_path, 'w') as f:
        json.dump(partition_map, f, indent=4)
    print(f"数据划分映射已保存到: {partition_map_path}")

    print("\n--- 环境创建完成！---")


if __name__ == '__main__':
    with open(project_root / "configs/config.yaml", 'r') as f:
        main_config = yaml.safe_load(f)

    create_and_save_environment(main_config)