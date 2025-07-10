# hetero-fedmeta/federated/simulator.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import json
from pathlib import Path

# 导入我们项目中的其他模块
from models.fl_model import SimpleCNN_CIFAR10, SimpleCNN_MNIST  # 【新】导入两个分离的模型
from federated.client import Client
from utils.data_partition import partition_data_non_iid
from federated.aggregation import get_aggregator


class FLSimulator:
    def __init__(self, config):
        """
        初始化模拟器。每次创建实例时，都会构建一个全新的、完整的模拟环境。
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模拟器使用设备: {self.device}")

        # 直接在初始化时加载并设置好所有环境
        self._setup_environment()

        # 【核心修改】根据配置选择正确的模型类来初始化全局模型
        dataset_name = self.config['fl']['dataset']
        if dataset_name == 'cifar10':
            self.global_model = SimpleCNN_CIFAR10(num_classes=config['fl']['num_classes']).to(self.device)
        elif dataset_name == 'mnist':
            self.global_model = SimpleCNN_MNIST(num_classes=config['fl']['num_classes']).to(self.device)
        else:
            raise ValueError(f"未知的数据集: {dataset_name}, 无法确定模型。")

        self.global_clock = 0.0
        self.results = []

        # 初始化统计计数器
        self.total_clients_selected = 0
        self.total_clients_dropped = 0

        # 根据配置创建聚合器实例
        self.aggregator = get_aggregator(config['fl']['aggregation_strategy'])

    def _setup_environment(self):
        """
        一次性设置好整个联邦学习环境，包括客户端创建和数据分配。
        """
        print("--- 正在创建/加载联邦学习环境 ---")
        project_root = Path(__file__).resolve().parents[1]

        # 【核心修改】根据所有相关配置构建正确的环境路径
        dataset_name = self.config['fl']['dataset']
        alpha = self.config['fl']['non_iid_alpha']
        num_clients = self.config['meta']['client_num']

        env_name = f"env_{dataset_name}_alpha{alpha}_clients{num_clients}"
        env_dir = project_root / "data" / env_name

        if not env_dir.exists():
            raise FileNotFoundError(
                f"错误: 找不到环境目录 {env_dir}。\n"
                f"请先在 configs/config.yaml 中设置好正确的参数 (dataset, alpha, client_num),\n"
                f"然后运行 'experiments/create_environment.py' 来创建它。"
            )

        # ... 后续加载文件的代码都从这个新的 env_dir 加载 ...

        # 1. 加载设备画像并创建客户端
        profiles_path = env_dir / "device_profiles.csv"
        print(f"加载设备画像: {profiles_path}")
        profile_df = pd.read_csv(profiles_path)

        self.client_pool = []
        for _, row in profile_df.iterrows():
            client_id = row['client_id']
            device_profile = row.iloc[1:].values
            self.client_pool.append(Client(client_id, device_profile))
        print(f"成功创建 {len(self.client_pool)} 个客户端。")

        # 2. 准备数据集
        dataset_name = self.config['fl']['dataset']
        print(f"准备数据集: {dataset_name.upper()}")

        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
            self.test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
            self.test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError(f"未知的数据集: {dataset_name}")

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['fl']['test_batch_size'], shuffle=False)

        # 3. 加载数据划分并分配给客户端
        partition_path = env_dir / "partition_map.json"
        print(f"加载数据划分: {partition_path}")
        with open(partition_path, 'r') as f:
            partition_map = json.load(f)

        if len(self.client_pool) != len(partition_map):
            print(f"警告: 客户端数量({len(self.client_pool)})与数据分片数量({len(partition_map)})不匹配！")

        for client_id_str, indices in partition_map.items():
            client_id = int(client_id_str)
            if client_id < len(self.client_pool):
                self.client_pool[client_id].local_data = Subset(self.train_dataset, indices)
                self.client_pool[client_id].data_size = len(indices)

        print("数据已成功分配给所有客户端。")

    def _client_local_training(self, client):
        """
        模拟单个客户端的本地训练过程。
        """
        # 【核心修改】本地模型也根据配置选择正确的类
        dataset_name = self.config['fl']['dataset']
        if dataset_name == 'cifar10':
            local_model = SimpleCNN_CIFAR10(num_classes=self.config['fl']['num_classes']).to(self.device)
        elif dataset_name == 'mnist':
            local_model = SimpleCNN_MNIST(num_classes=self.config['fl']['num_classes']).to(self.device)
        else:
            raise ValueError(f"未知的数据集: {dataset_name}, 无法确定模型。")

        local_model.load_state_dict(self.global_model.state_dict())
        local_model.train()

        train_loader = DataLoader(client.local_data, batch_size=32, shuffle=True)
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.config['meta']['base_epochs']):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return local_model.state_dict()

    def _select_clients(self, available_clients, strategy='random'):
        """
        根据指定策略选择客户端。
        """
        num_to_select = self.config['meta']['max_clients']
        if len(available_clients) <= num_to_select:
            return available_clients

        if strategy == 'random':
            return random.sample(available_clients, num_to_select)
        elif strategy == 'resource_first':
            available_clients.sort(key=lambda c: c.power + c.bandwidth, reverse=True)
            return available_clients[:num_to_select]
        else:
            raise ValueError(f"未知的选择策略: {strategy}")

    def run(self, selection_strategy='random'):
        """
        运行完整的联邦学习模拟。
        """
        print(
            f"\n--- 开始联邦学习模拟 (选择策略: {selection_strategy}, 聚合策略: {self.config['fl']['aggregation_strategy']}) ---")

        for r in range(self.config['meta']['max_rounds']):
            print(f"\n>>>>> 全局轮次 {r + 1}/{self.config['meta']['max_rounds']} <<<<<")

            available_clients = [c for c in self.client_pool if c.is_available()]
            print(f"本轮共有 {len(available_clients)}/{len(self.client_pool)} 个客户端可用。")

            selected_clients = self._select_clients(available_clients, strategy=selection_strategy)
            print(f"选择了 {len(selected_clients)} 个客户端参与本轮训练。")
            self.total_clients_selected += len(selected_clients)

            client_results = []
            # 【新增】用于统计本轮掉线数的计数器
            clients_dropped_this_round = 0

            for client in tqdm(selected_clients, desc="模拟客户端训练"):
                if not client.check_if_online():
                    self.total_clients_dropped += 1
                    clients_dropped_this_round += 1  # 【新增】本轮掉线数+1
                    continue

                update = self._client_local_training(client)
                training_time = client.get_training_time()
                comm_time = client.get_communication_time(model_size_mb=self.config['fl']['model_size'])

                client_results.append({
                    'update': update,
                    'time': training_time + comm_time,
                    'data_size': client.data_size
                })

            # 【新增】打印本轮的掉线统计信息
            num_successful = len(client_results)
            print(f"本轮训练完成: {num_successful} 个成功, {clients_dropped_this_round} 个掉线。")

            if not client_results:
                print("本轮没有客户端成功完成训练，跳过聚合。")
                self.evaluate_and_log(r)
                continue

            round_times = [res['time'] for res in client_results]
            time_cost_this_round = max(round_times)
            self.global_clock += time_cost_this_round
            print(f"本轮耗时 (由最慢的客户端决定): {time_cost_this_round:.2f} 秒。")
            print(f"累计模拟时间: {self.global_clock:.2f} 秒。")

            print("正在调用聚合器...")
            new_global_state_dict = self.aggregator.aggregate(
                self.global_model.state_dict(),
                client_results
            )
            self.global_model.load_state_dict(new_global_state_dict)

            self.evaluate_and_log(r)

        print("\n--- 联邦学习模拟结束 ---")
        self.print_final_stats(selection_strategy)
        return pd.DataFrame(self.results)

    def evaluate_and_log(self, current_round):
        """封装评估和记录的逻辑"""
        accuracy = self.evaluate_global_model()
        print(f"本轮结束后，全局模型准确率: {accuracy:.4f}")
        self.results.append({'round': current_round + 1, 'time': self.global_clock, 'accuracy': accuracy})

    def print_final_stats(self, selection_strategy):
        """打印最终的统计信息"""
        if self.total_clients_selected > 0:
            overall_dropout_rate = (self.total_clients_dropped / self.total_clients_selected) * 100
            print(f"\n--- 策略 '{selection_strategy}' 的总体统计 ---")
            print(f"总共选择了 {self.total_clients_selected} 次客户端参与训练。")
            print(f"其中，总共有 {self.total_clients_dropped} 次掉线。")
            print(f"总体掉线率: {overall_dropout_rate:.2f}%")
        else:
            print("没有客户端被选择，无法计算掉线率。")

    def evaluate_global_model(self):
        """在测试集上评估当前全局模型的性能"""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        self.global_model.train()
        return accuracy