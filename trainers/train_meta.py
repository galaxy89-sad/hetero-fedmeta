# trainers/train_meta.py
import torch
from torch.utils.data import DataLoader, Subset
from federated.client import HeterogeneousClient
from data.dataset import DatasetHandler


def generate_meta_tasks(config, generator, num_tasks):
    """生成元学习任务，每个任务包含一个异构客户端及其特征"""
    dataset_handler = DatasetHandler(config)
    train_data, _ = dataset_handler.load_dataset()
    tasks = []

    for i in range(num_tasks):
        # 从生成器获取设备和数据特征
        z = torch.randn(1, config['gan']['latent_dim'])
        with torch.no_grad():
            device_features, data_features = generator(z)

        # 提取特征值
        device_features = device_features.squeeze(0)
        data_features = data_features.squeeze(0)

        # 根据生成的数据特征中的样本量创建数据子集
        # 注意：这里需要确保生成的样本量不超过原始数据集大小
        data_size = int(data_features[0].item())
        data_size = min(data_size, len(train_data))

        # 随机选择样本
        indices = torch.randperm(len(train_data))[:data_size].tolist()
        client_subset = Subset(train_data, indices)

        # 创建客户端（使用生成的特征）
        client = HeterogeneousClient(
            client_id=i,
            client_subset=client_subset,
            device_features=device_features,
            data_features=data_features,
            config=config
        )

        # 创建元学习任务
        task = {
            'client': client,
            'device_features': device_features,
            'data_features': data_features
        }
        tasks.append(task)

    return tasks