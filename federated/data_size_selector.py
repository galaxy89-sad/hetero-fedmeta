# federated/data_size_selector.py
from .selector_base import SelectorBase
import numpy as np

class DataSizeSelector(SelectorBase):
    def select_clients(self, client_features, num_clients):
        """选择数据量最大的客户端"""
        # 提取数据量（假设特征第4位是数据量）
        data_sizes = [feat[3] for feat in client_features]
        # 按数据量降序排序，取前num_clients个
        sorted_indices = np.argsort(data_sizes)[::-1]
        return sorted_indices[:num_clients]