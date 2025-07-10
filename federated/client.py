# hetero-fedmeta/federated/client.py

import numpy as np


class Client:
    """
    模拟一个具有异构特性的联邦学习客户端。
    这个类的实例代表一个独立的设备，其行为由唯一的设备画像驱动。
    """

    def __init__(self, client_id, device_profile):
        """
        初始化一个客户端。

        参数:
        - client_id (int): 客户端的唯一ID。
        - device_profile (np.array): 一个5维的归一化设备画像向量
                                     [power, bandwidth, ram, stability, energy]。
        """
        self.client_id = client_id

        # 解包设备画像，并赋予明确含义
        self.power = device_profile[0]
        self.bandwidth = device_profile[1]
        self.ram = device_profile[2]
        self.stability = device_profile[3]
        self.energy = device_profile[4]

        # 本地数据集，初始化为空，将在后续步骤中分配
        self.local_data_indices = None
        self.data_size = 0

    def __repr__(self):
        """定义对象的字符串表示，方便调试"""
        return (f"Client(ID={self.client_id}, "
                f"Power={self.power:.2f}, "
                f"Bandwidth={self.bandwidth:.2f}, "
                f"Stability={self.stability:.2f}, "
                f"Energy={self.energy:.2f}, "
                f"DataSize={self.data_size})")

    def get_training_time(self, base_training_time=10.0, min_power=0.01):
        """
        根据算力(power)计算本地训练时长。
        算力越高，训练时间越短。
        参数:
        - base_training_time (float): 一个标准设备完成一轮训练所需的基础时间（秒）。
        - min_power (float): 一个很小的数，防止除以零。
        返回:
        - float: 模拟的训练时间（秒）。
        """
        # 算力与时间成反比
        return base_training_time / (self.power + min_power)

    def get_communication_time(self, model_size_mb=10.0, min_bandwidth=0.01):
        """
        根据带宽(bandwidth)计算模型上传/下载的通信时长。
        带宽越高，通信时间越短。
        参数:
        - model_size_mb (float): 要传输的模型大小（MB）。
        - min_bandwidth (float): 一个很小的数，防止除以零。
        返回:
        - float: 模拟的通信时间（秒）。
        """
        # 带宽与时间成反比
        # 假设带宽单位是 Mbps (Megabits per second), 模型大小是 MB (Megabytes)
        # 1 MB = 8 Mbits
        model_size_mbits = model_size_mb * 8
        # 带宽值为[0,1]，我们需要将其映射到一个实际的速率范围，例如[1, 100] Mbps
        # 这里简化处理，直接使用归一化值，但需注意其相对意义
        # 实际带宽 = 1 + self.bandwidth * 99 (映射到1-100Mbps)
        # 此处简化为直接反比关系，重点是体现相对快慢
        return model_size_mbits / (self.bandwidth * 100 + min_bandwidth)

    def check_if_online(self):
        """
        根据稳定性(stability)模拟客户端是否掉线。
        稳定性越高，成功在线的概率越大。
        返回:
        - bool: True表示在线，False表示掉线。
        """
        # stability值直接作为成功在线的概率
        return np.random.rand() <= self.stability
        # 对比无掉线率
        # return True

    def is_available(self, energy_threshold=0.9):
        """
        根据能耗压力(energy)判断客户端是否可用。
        能耗压力过大（电量过低）则不可用。
        参数:
        - energy_threshold (float): 能耗压力的上限阈值。
        返回:
        - bool: True表示可用，False表示不可用。
        """
        # energy值越高代表情况越差，所以如果超过阈值，则不可用
        return self.energy < energy_threshold