# hetero-fedmeta/models/sc_gan.py

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    生成器 (Generator)
    职责: 接收一个随机噪声向量，输出一个5维的、看起来像真实设备画像的向量。
    """

    def __init__(self, noise_dim=100, output_dim=5):
        """
        初始化生成器网络结构。

        参数:
        - noise_dim (int): 输入的随机噪声向量的维度。
        - output_dim (int): 输出的设备画像向量的维度（在我们的案例中是5）。
        """
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # 第一层：将噪声向量映射到更大的维度空间
            nn.Linear(noise_dim, 128),
            nn.ReLU(inplace=True),  # 使用ReLU激活函数增加非线性

            # 第二层：进一步扩大维度空间
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),

            # 输出层：映射到最终的5维向量
            nn.Linear(256, output_dim),

            # 关键！使用Sigmoid激活函数将输出值严格限制在[0, 1]区间。
            # 这与我们已经归一化的数据完美匹配。
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        前向传播。

        参数:
        - z (torch.Tensor): 输入的随机噪声，形状为 (batch_size, noise_dim)。

        返回:
        - torch.Tensor: 生成的设备画像向量，形状为 (batch_size, output_dim)。
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    判别器 (Discriminator)
    职责: 接收一个5维的设备画像向量，判断它是真实的（来自数据集）还是伪造的（来自生成器）。
    """

    def __init__(self, input_dim=5):
        """
        初始化判别器网络结构。

        参数:
        - input_dim (int): 输入的设备画像向量的维度（在我们的案例中是5）。
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 第一层：将5维输入映射到更大的特征空间
            nn.Linear(input_dim, 256),
            # 使用LeakyReLU可以防止梯度消失问题，在GAN中是常见技巧
            nn.LeakyReLU(0.2, inplace=True),

            # 第二层：进一步提取特征
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层：输出一个单一的数值（logit）
            # 这个值代表输入是“真实”的置信度。
            # 这里不加Sigmoid，因为我们将使用BCEWithLogitsLoss，它更数值稳定。
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        前向传播。

        参数:
        - x (torch.Tensor): 输入的设备画像向量，形状为 (batch_size, input_dim)。

        返回:
        - torch.Tensor: 判别器输出的logit，形状为 (batch_size, 1)。
        """
        return self.model(x)