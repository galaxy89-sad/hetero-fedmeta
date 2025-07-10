# federated/random_selector.py
from .selector_base import SelectorBase
import numpy as np

class RandomSelector(SelectorBase):
    def select_clients(self, client_features, num_clients):
        """完全随机选择客户端"""
        total_clients = len(client_features)
        return np.random.choice(total_clients, size=num_clients, replace=False)