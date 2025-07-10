import torch
import numpy as np
from torch.utils.data import DataLoader


def split_data_by_distribution(dataset, distribution, num_samples=None):
    """根据分布划分数据集"""
    if num_samples is None:
        num_samples = len(dataset)

    # 获取数据集标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # 如果数据集没有targets属性，手动获取
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # 按类别分组数据索引
    class_indices = [np.where(labels == c)[0] for c in range(len(distribution))]

    # 根据分布采样
    indices = []
    for c, prob in enumerate(distribution):
        num_c = int(num_samples * prob)
        # 确保不超过该类别的样本数
        num_c = min(num_c, len(class_indices[c]))
        indices.extend(np.random.choice(class_indices[c], num_c, replace=False))

    return indices


def get_data_distribution(dataset):
    """获取数据集的标签分布"""
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    return np.bincount(labels) / len(labels)