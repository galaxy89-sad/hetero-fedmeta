import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class DatasetHandler:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['fl']['dataset']
        self.num_classes = config['fl']['num_classes']
        self.non_iid_alpha = config['fl']['non_iid_alpha']

    def load_dataset(self):
        """加载数据集并返回训练集、测试集"""
        if self.dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10('./data', train=False, transform=transform)
        elif self.dataset_name == "mnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_data = datasets.MNIST('./data', train=False, transform=transform)
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        return train_data, test_data

    def split_non_iid(self, train_data, num_clients):
        """按Dirichlet分布划分Non-IID数据"""
        labels = np.array(train_data.targets)
        client_distributions = np.random.dirichlet([self.non_iid_alpha] * self.num_classes, num_clients)
        class_indices = [np.where(labels == c)[0] for c in range(self.num_classes)]

        client_data = []
        for i in range(num_clients):
            indices = []
            for c in range(self.num_classes):
                n = int(len(train_data) * client_distributions[i, c])
                n = min(n, len(class_indices[c]))
                if n > 0:
                    indices.extend(np.random.choice(class_indices[c], n, replace=False))
            client_data.append(Subset(train_data, indices))

        return client_data, client_distributions
