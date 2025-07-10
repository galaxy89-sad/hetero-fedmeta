# federated/selector_base.py
import numpy as np

class SelectorBase:
    def select_clients(self, client_features, num_clients):
        """
        选择客户端的统一接口
        :param client_features: 客户端特征列表（每个元素是客户端的异构特征向量）
        :param num_clients: 每轮要选择的客户端数量
        :return: 选中的客户端索引列表
        """
        raise NotImplementedError("子类必须实现 select_clients 方法")