# hetero-fedmeta/utils/device_factory.py

import torch
import numpy as np
from pathlib import Path

# 导入GAN生成器模型架构和我们刚定义的Client类
from models.sc_gan import Generator
from federated.client import Client


def load_generator(checkpoint_path, noise_dim=100, output_dim=5):
    """
    加载训练好的GAN生成器模型。

    参数:
    - checkpoint_path (str or Path): 指向生成器模型权重文件 (.pth) 的路径。
    - noise_dim (int): 生成器输入噪声的维度。
    - output_dim (int): 生成器输出画像的维度。

    返回:
    - torch.nn.Module: 加载了权重的生成器模型，并移动到合适的设备。
    """
    # 确定设备 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化与训练时相同架构的模型
    generator = Generator(noise_dim, output_dim).to(device)

    # 加载状态字典（模型权重）
    # 使用 map_location=device 确保模型能被正确加载到当前设备，无论当初是在CPU还是GPU上保存的
    print(f"正在从 {checkpoint_path} 加载生成器模型...")
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 将模型设置为评估模式，这会关闭Dropout等训练时才需要的层
    generator.eval()

    print("生成器模型加载成功！")
    return generator


def create_client_population(num_clients, generator):
    """
    使用训练好的生成器创建指定数量的客户端实例。

    参数:
    - num_clients (int): 要创建的客户端总数。
    - generator (torch.nn.Module): 已经加载并设为评估模式的GAN生成器。

    返回:
    - list[Client]: 一个包含所有创建好的客户端实例的列表。
    """
    print(f"正在创建 {num_clients} 个模拟客户端...")

    # 确定设备
    device = next(generator.parameters()).device

    # 一次性生成所有需要的噪声向量
    noise = torch.randn(num_clients, generator.model[0].in_features).to(device)

    # 使用 no_grad() 上下文管理器，因为我们只是在做推理，不需要计算梯度
    with torch.no_grad():
        # 生成所有设备画像
        device_profiles = generator(noise).cpu().numpy()

    # 为每个生成的画像创建一个Client实例
    client_population = []
    for i in range(num_clients):
        client = Client(client_id=i, device_profile=device_profiles[i])
        client_population.append(client)

    print(f"成功创建 {len(client_population)} 个客户端。")
    return client_population


# --- 这是一个使用示例，可以放在main.py或你的主模拟脚本中 ---
if __name__ == '__main__':
    # 这是一个演示如何使用这些函数的例子

    # 1. 定义路径和参数
    # 在实际项目中，这些路径和参数应该从config.yaml中读取
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/sc_gan/generator_epoch_200.pth"
    NUM_CLIENTS = 100  # 假设我们要创建100个客户端

    # 2. 加载生成器
    gan_generator = load_generator(CHECKPOINT_PATH)

    # 3. 创建客户端总体
    all_clients = create_client_population(num_clients=NUM_CLIENTS, generator=gan_generator)

    # 4. 查看一些创建好的客户端
    print("\n--- 创建的客户端示例 ---")
    for i in range(5):
        print(all_clients[i])

    # 5. 模拟一个客户端的行为
    print("\n--- 模拟Client 0的行为 ---")
    client_0 = all_clients[0]
    training_time = client_0.get_training_time()
    comm_time = client_0.get_communication_time()
    is_available = client_0.is_available()
    is_online = client_0.check_if_online()

    print(f"Client 0 是否可用: {is_available}")
    if is_available:
        print(f"  - 模拟训练时间: {training_time:.2f} 秒")
        print(f"  - 模拟通信时间: {comm_time:.2f} 秒")
        print(f"  - 本轮是否成功在线: {is_online}")