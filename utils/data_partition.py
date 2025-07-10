# hetero-fedmeta/utils/data_partition.py

import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset


def partition_data_non_iid(dataset, num_clients, alpha=0.5):
    """
    使用狄利克雷分布将数据集划分为Non-IID的分片。

    参数:
    - dataset (torch.utils.data.Dataset): 完整的原始数据集 (例如, CIFAR10)。
    - num_clients (int): 客户端的总数。
    - alpha (float): 狄利克雷分布的参数。alpha越小，数据异构性越强。

    返回:
    - list[torch.utils.data.Subset]: 一个列表，每个元素都是一个客户端的本地数据集（作为Subset）。
    """
    print(f"正在以 Non-IID (alpha={alpha}) 方式为 {num_clients} 个客户端划分数据...")

    # 获取数据集的标签
    # 注意: dataset.targets 对于CIFAR10是有效的，对于其他数据集可能需要调整
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        # 兼容其他类型的数据集
        labels = np.array([sample[1] for sample in dataset])

    num_classes = len(np.unique(labels))
    num_samples = len(dataset)

    # 记录每个类别有哪些样本索引
    indices_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # 为每个客户端生成一个狄利克雷分布的标签比例
    # (num_clients, num_classes)
    label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)

    # 分配每个客户端每个类别的样本数量
    # 首先，确保每个客户端至少有一个样本，避免空数据集
    client_indices = [[] for _ in range(num_clients)]

    # 遍历每个类别，按比例分配索引
    for c, indices in indices_by_class.items():
        # 获取该类别在此次划分中的样本比例
        proportions = label_distribution[:, c]
        proportions = proportions / proportions.sum()  # 归一化，确保总和为1

        # 计算每个客户端应该分配多少个该类别的样本
        samples_per_client = (proportions * len(indices)).astype(int)

        # 处理取整误差，将剩余样本分配给比例最大的客户端
        remainder = len(indices) - samples_per_client.sum()
        if remainder > 0:
            add_where = np.random.choice(num_clients, size=remainder, p=proportions)
            for i in add_where:
                samples_per_client[i] += 1

        # 从该类别的索引池中分配索引
        current_pos = 0
        for client_id in range(num_clients):
            num_to_take = samples_per_client[client_id]
            if num_to_take > 0:
                client_indices[client_id].extend(indices[current_pos: current_pos + num_to_take])
                current_pos += num_to_take

    # 创建Subset对象列表
    client_datasets = [Subset(dataset, indices) for indices in client_indices]

    print("数据划分完成！")
    return client_datasets