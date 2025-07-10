# hetero-fedmeta/trainers/train_sc_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import os

# 导入我们刚刚定义的模型架构
# 使用相对导入，假设脚本从项目根目录运行或已正确设置PYTHONPATH
from models.sc_gan import Generator, Discriminator

# --- 配置参数 ---
# 在实际项目中，这些参数应该从 configs/config.yaml 文件中加载
# 这里为了方便直接定义
CONFIG = {
    "epochs": 200,  # 训练的总轮次
    "batch_size": 64,  # 每批次的数据量
    "lr": 0.0002,  # 学习率
    "beta1": 0.5,  # Adam优化器的beta1参数，0.5是GAN训练的常用值
    "noise_dim": 100,  # 噪声向量的维度
    "data_path": "data/normalized_device_profiles_v3.csv",  # 训练数据路径
    "checkpoint_dir": "checkpoints/sc_gan/",  # 模型保存路径
    "output_dim": 5  # 设备画像的维度
}


class GANTrainer:
    """
    GAN训练器
    封装了整个GAN的训练、评估和模型保存逻辑。
    """

    def __init__(self, config):
        self.config = config

        # 获取项目根目录，以便于处理路径
        self.project_root = Path(__file__).resolve().parents[1]

        # 设置设备 (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化模型
        self.generator = Generator(self.config["noise_dim"], self.config["output_dim"]).to(self.device)
        self.discriminator = Discriminator(self.config["output_dim"]).to(self.device)

        # 初始化优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.config["lr"],
                                      betas=(self.config["beta1"], 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.config["lr"],
                                      betas=(self.config["beta1"], 0.999))

        # 损失函数 (BCEWithLogitsLoss比BCELoss+Sigmoid更数值稳定)
        self.criterion = nn.BCEWithLogitsLoss()

        # 准备数据
        self.dataloader = self._prepare_data()

        # 原始数据统计信息，用于评估
        self.real_data_df = pd.read_csv(self.project_root / self.config["data_path"])

    def _prepare_data(self):
        """加载并准备数据加载器"""
        data_path = self.project_root / self.config["data_path"]
        print(f"从 {data_path} 加载数据...")

        df = pd.read_csv(data_path)
        # 将数据转换为PyTorch Tensor
        tensors = torch.tensor(df.values, dtype=torch.float32)
        dataset = TensorDataset(tensors)

        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, drop_last=True)
        return dataloader

    def train(self):
        """执行完整的训练流程"""
        print("开始训练GAN...")
        for epoch in range(self.config["epochs"]):
            # 使用tqdm显示进度条
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                desc=f"Epoch {epoch + 1}/{self.config['epochs']}")

            for i, (real_samples,) in progress_bar:
                real_samples = real_samples.to(self.device)
                batch_size = real_samples.size(0)

                # 定义真实和虚假的标签
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # ---------------------
                #  训练判别器 (Discriminator)
                # ---------------------
                self.d_optimizer.zero_grad()

                # 1. 计算判别器在真实数据上的损失
                d_output_real = self.discriminator(real_samples)
                d_loss_real = self.criterion(d_output_real, real_labels)

                # 2. 生成虚假数据并计算判别器在虚假数据上的损失
                noise = torch.randn(batch_size, self.config["noise_dim"]).to(self.device)
                fake_samples = self.generator(noise)
                d_output_fake = self.discriminator(fake_samples.detach())  # detach() 避免梯度传到生成器
                d_loss_fake = self.criterion(d_output_fake, fake_labels)

                # 3. 总损失并更新判别器
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # -----------------
                #  训练生成器 (Generator)
                # -----------------
                self.g_optimizer.zero_grad()

                # 我们希望生成器生成的假样本能被判别器认为是"真实"的
                # 所以我们用real_labels (全1)来计算生成器的损失
                d_output_g = self.discriminator(fake_samples)  # 注意这里fake_samples没有detach
                g_loss = self.criterion(d_output_g, real_labels)

                # 更新生成器
                g_loss.backward()
                self.g_optimizer.step()

                # 更新进度条显示
                progress_bar.set_postfix({"D Loss": f"{d_loss.item():.4f}", "G Loss": f"{g_loss.item():.4f}"})

            # 在每个epoch结束时进行评估和保存
            if (epoch + 1) % 10 == 0 or epoch == self.config["epochs"] - 1:
                self.evaluate_and_save(epoch + 1)

        print("训练完成！")

    def evaluate_and_save(self, epoch):
        """评估模型并保存检查点"""
        print(f"\n--- Epoch {epoch} 评估 ---")

        # 将模型设为评估模式
        self.generator.eval()

        with torch.no_grad():
            # 生成大量样本用于评估
            noise = torch.randn(len(self.real_data_df), self.config["noise_dim"]).to(self.device)
            generated_samples = self.generator(noise).cpu().numpy()
            generated_df = pd.DataFrame(generated_samples, columns=self.real_data_df.columns)

            print("真实数据统计信息:")
            print(self.real_data_df.describe().round(4))
            print("\n生成数据统计信息:")
            print(generated_df.describe().round(4))

            print("\n真实数据相关性矩阵:")
            print(self.real_data_df.corr().round(4))
            print("\n生成数据相关性矩阵:")
            print(generated_df.corr().round(4))

        # 将模型恢复为训练模式
        self.generator.train()

        # 保存模型检查点
        checkpoint_path = self.project_root / self.config["checkpoint_dir"]
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(self.generator.state_dict(), checkpoint_path / f"generator_epoch_{epoch}.pth")
        torch.save(self.discriminator.state_dict(), checkpoint_path / f"discriminator_epoch_{epoch}.pth")
        print(f"模型已保存到 {checkpoint_path}")
        print("--- 评估结束 ---\n")


if __name__ == '__main__':
    # 创建并启动训练器
    trainer = GANTrainer(CONFIG)
    trainer.train()