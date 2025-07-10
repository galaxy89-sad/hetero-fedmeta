import numpy as np


class ClientSelector:
    def __init__(self, method="random", meta_learner=None):
        self.method = method
        self.meta_learner = meta_learner

    def select(self, clients, num_clients=20):
        if self.method == "random":
            return np.random.choice(len(clients), num_clients, replace=False)

        elif self.method == "data_size":
            # 按数据量大小选择
            data_sizes = [client.data_size for client in clients]
            return np.argsort(data_sizes)[-num_clients:]

        elif self.method == "compute_power":
            # 按计算能力选择
            compute_powers = [client.compute_power for client in clients]
            return np.argsort(compute_powers)[-num_clients:]

        elif self.method == "meta":
            # 使用元学习选择
            if self.meta_learner is None:
                raise ValueError("Meta-learner is required for meta selection method")
            return self.meta_learner.select_clients(clients, num_clients)

        else:
            raise ValueError(f"Unsupported selection method: {self.method}")