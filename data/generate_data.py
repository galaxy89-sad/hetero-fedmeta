import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader


def load_dataset(name="cifar10"):
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {name} not supported")

    return train_data, test_data, num_classes


def split_data_non_iid(train_data, num_clients, alpha=0.1):
    """按Dirichlet分布划分Non-IID数据"""
    num_classes = len(np.unique(train_data.targets))
    labels = np.array(train_data.targets)

    # 为每个客户端生成标签分布
    client_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)

    # 按标签分组数据
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    # 为每个客户端分配数据
    client_data_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        # 打乱类别c的数据
        np.random.shuffle(class_indices[c])

        # 按分布划分给客户端
        proportions = client_distributions[:, c]
        proportions = proportions / proportions.sum()  # 归一化
        split_points = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        client_splits = np.split(class_indices[c], split_points)

        # 分配给客户端
        for i in range(num_clients):
            client_data_indices[i].extend(client_splits[i])

    # 创建客户端数据集
    client_datasets = [Subset(train_data, indices) for indices in client_data_indices]

    return client_datasets, client_distributions


def split_data_by_distribution(dataset, distribution, num_samples=None):
    """
    根据类别分布划分数据集（支持IID/Non-IID划分）
    :param dataset: PyTorch数据集对象
    :param distribution: 类别概率分布（长度为num_classes的列表）
    :param num_samples: 划分的样本总数（默认取数据集全部样本）
    :return: 划分后的Subset对象
    """
    if num_samples is None:
        num_samples = len(dataset)

    # 提取标签（兼容不同数据集格式）
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])  # 遍历取标签（慢，适合小数据）

    # 按类别分组索引
    num_classes = len(distribution)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    selected_indices = []
    for c in range(num_classes):
        # 计算该类别应选样本数（避免超过实际数量）
        target = int(num_samples * distribution[c])
        available = len(class_indices[c])
        select_num = min(target, available)

        # 随机选择索引
        if select_num > 0:
            selected = np.random.choice(class_indices[c], select_num, replace=False)
            selected_indices.extend(selected)

    return Subset(dataset, selected_indices)