import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data.dataset import split_data_by_distribution  # 确保导入数据划分函数


class ClientConfig:
    def __init__(
            self,
            client_id,
            data_loader,
            compute_power=1.0,
            bandwidth=10.0,
            memory_limit=1024,
            data_distribution=None,
            stability=1.0,
            energy_cost=1.0
    ):
        self.client_id = client_id
        self.data_loader = data_loader
        self.compute_power = compute_power
        self.bandwidth = bandwidth
        self.memory_limit = memory_limit
        self.data_distribution = data_distribution or np.ones(10) / 10  # 默认均匀分布
        self.stability = stability  # 在线率（0-1）
        self.energy_cost = energy_cost  # 能耗系数
        self.data_size = len(data_loader.dataset)  # 本地数据量

    def get_local_training_time(self, base_epochs, model_size):
        """计算本地训练+上传总时间（秒）"""
        compute_time = base_epochs * self.data_size / 1000 / self.compute_power  # 计算时间
        upload_time = (model_size * 8) / (self.bandwidth * 10**6)  # 上传时间（Mbps→bit/s）
        return compute_time + upload_time

    def get_upload_cost(self, model_size):
        """单独计算上传时间（秒）"""
        return (model_size * 8) / (self.bandwidth * 10**6)

    def is_online(self):
        """模拟客户端在线状态"""
        return np.random.rand() < self.stability


def simulate_client_training(client_config, global_model, base_epochs, model_size, device=torch.device("cpu")):
    """模拟客户端本地训练（考虑异构性）"""
    if not client_config.is_online():
        return None, client_config.get_local_training_time(0, model_size)

    local_model = copy.deepcopy(global_model).to(device)
    adjusted_epochs = max(1, int(base_epochs * client_config.compute_power))  # 调整训练轮次

    # 内存检查（简化模拟）
    model_memory = get_model_memory_size(local_model)
    if model_memory > client_config.memory_limit:
        return None, client_config.get_local_training_time(0, model_size)

    # 本地训练
    optimizer = optim.SGD(local_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(adjusted_epochs):
        for inputs, labels in client_config.data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    actual_train_time = time.time() - start_time
    simulated_time = client_config.get_local_training_time(adjusted_epochs, model_size)
    return local_model, simulated_time


def get_model_memory_size(model):
    """计算模型内存占用（MB）"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 * 1024)  # 浮点数占4字节


def generate_heterogeneous_clients(num_clients, train_data, num_classes, batch_size=32, non_iid_alpha=0.1):
    """生成异构客户端集合（增强参数验证）"""
    # 验证并转换 num_classes 为整数
    try:
        num_classes = int(num_classes)
        assert num_classes > 0, "num_classes 必须是正整数"
    except (ValueError, TypeError, AssertionError) as e:
        raise ValueError(f"无效的 num_classes: {num_classes}. 错误: {str(e)}")

    # 验证并转换 non_iid_alpha 为浮点数
    try:
        non_iid_alpha = float(non_iid_alpha)
        assert non_iid_alpha > 0, "non_iid_alpha 必须大于0"
    except (ValueError, TypeError, AssertionError) as e:
        raise ValueError(f"无效的 non_iid_alpha: {non_iid_alpha}. 错误: {str(e)}")

    clients = []
    for client_id in range(num_clients):
        # 资源配置
        compute_power = np.random.uniform(0.5, 2.0)
        bandwidth = np.random.uniform(1.0, 100.0)
        memory_limit = np.random.randint(256, 4097)

        # 生成Dirichlet分布
        alpha = np.full(num_classes, non_iid_alpha, dtype=np.float64)
        distribution = np.random.dirichlet(alpha)

        # 划分数据并创建客户端
        client_subset = split_data_by_distribution(train_data, distribution)
        data_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)

        client = ClientConfig(
            client_id=client_id,
            data_loader=data_loader,
            compute_power=compute_power,
            bandwidth=bandwidth,
            memory_limit=memory_limit,
            data_distribution=distribution
        )
        clients.append(client)

    return clients