# hetero-fedmeta/federated/aggregation.py

import torch
from abc import ABC, abstractmethod
from collections import OrderedDict


class Aggregator(ABC):
    """
    聚合器策略的抽象基类 (Abstract Base Class)。
    所有具体的聚合方法都应继承自这个类。
    """

    @abstractmethod
    def aggregate(self, global_model_state_dict, client_updates):
        """
        定义如何将客户端更新聚合到全局模型中。

        参数:
        - global_model_state_dict (OrderedDict): 当前的全局模型参数字典。
        - client_updates (list[dict]): 一个列表，包含了所有成功返回的客户端更新。
                                        每个元素是一个字典，至少包含 'update' (模型参数) 和 'data_size'。

        返回:
        - OrderedDict: 聚合后的新全局模型参数字典。
        """
        pass


class FedAvg(Aggregator):
    """
    朴素联邦平均 (Simple Averaging)。
    所有客户端的贡献权重相等。
    """

    def aggregate(self, global_model_state_dict, client_updates):
        print("使用 FedAvg (简单平均) 聚合策略...")

        # 如果没有有效的更新，直接返回原模型
        if not client_updates:
            return global_model_state_dict

        # 复制一份全局模型状态，用于存放聚合结果
        avg_state_dict = OrderedDict(global_model_state_dict)

        for key in avg_state_dict.keys():
            # 将所有客户端在这一层的参数堆叠起来，然后计算平均值
            avg_state_dict[key] = torch.stack(
                [update['update'][key] for update in client_updates]
            ).mean(0)

        return avg_state_dict


class WeightedFedAvg(Aggregator):
    """
    加权联邦平均 (Weighted Averaging)。
    每个客户端的贡献根据其本地数据集的大小进行加权。
    """

    def aggregate(self, global_model_state_dict, client_updates):
        print("使用 WeightedFedAvg (加权平均) 聚合策略...")

        if not client_updates:
            return global_model_state_dict

        # 计算参与本轮聚合的总样本数
        total_data_size = sum(update['data_size'] for update in client_updates)

        # 如果总样本数为0，退化为简单平均或直接返回
        if total_data_size == 0:
            return FedAvg().aggregate(global_model_state_dict, client_updates)

        avg_state_dict = OrderedDict(global_model_state_dict)

        for key in avg_state_dict.keys():
            # 对每个客户端的参数进行加权求和
            weighted_sum = torch.stack(
                [update['update'][key] * (update['data_size'] / total_data_size) for update in client_updates]
            ).sum(0)
            avg_state_dict[key] = weighted_sum

        return avg_state_dict


# --- 工厂函数，用于根据配置动态选择聚合器 ---

def get_aggregator(name):
    """
    根据名称返回一个聚合器实例。
    """
    if name.lower() == 'fedavg':
        return FedAvg()
    elif name.lower() == 'weightedfedavg':
        return WeightedFedAvg()
    # 未来可以添加更多聚合器，例如:
    # elif name.lower() == 'fedprox':
    #     return FedProxAggregator()
    else:
        raise ValueError(f"未知的聚合器名称: {name}")