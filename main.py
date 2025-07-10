# hetero-fedmeta/main.py (修改后的版本)

import yaml
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime  # 导入datetime库来创建时间戳

# 导入我们项目中的模块
# 注意：我们不再需要 load_generator，因为 simulator 会自己处理
from federated.simulator import FLSimulator


def load_config(path="configs/config.yaml"):
    """加载YAML配置文件"""
    # 确保路径是相对于项目根目录的
    project_root = Path(__file__).resolve().parent
    with open(project_root / path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def plot_results(results_df_dict, output_dir):
    """
    可视化对比不同策略的结果，并将图片保存到指定的输出目录。

    参数:
    - results_df_dict (dict): 包含策略名称和对应结果DataFrame的字典。
    - output_dir (Path): 结果要保存到的文件夹路径。
    """
    plt.figure(figsize=(12, 5))

    # 图1: 准确率 vs. 轮次
    plt.subplot(1, 2, 1)
    for strategy, df in results_df_dict.items():
        plt.plot(df['round'], df['accuracy'], marker='o', linestyle='-', label=strategy)
    plt.title('Accuracy vs. Rounds')
    plt.xlabel('Global Round')
    plt.ylabel('Global Model Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # 图2: 准确率 vs. 时间
    plt.subplot(1, 2, 2)
    for strategy, df in results_df_dict.items():
        plt.plot(df['time'], df['accuracy'], marker='x', linestyle='--', label=strategy)
    plt.title('Accuracy vs. Time')
    plt.xlabel('Total Elapsed Time (seconds)')
    plt.ylabel('Global Model Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # 【修改】将图片保存到指定的输出目录
    output_path = output_dir / "fl_simulation_results.png"
    plt.savefig(output_path)
    print(f"结果图已保存到: {output_path}")
    plt.show()


if __name__ == '__main__':
    # --- 1. 设置实验环境 ---
    config = load_config()

    # a. 创建一个唯一的实验ID (基于当前日期和时间)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"开始新的实验运行，ID: {run_id}")

    # b. 创建本次实验的结果输出目录
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"所有结果将保存在: {output_dir}")

    # --- 2. 运行不同策略的模拟 ---
    results_dict = {}

    # 只创建一个模拟器实例，通过reset()来重置状态，保证公平性
    # (这里我们还是用创建新实例的方式，因为它更简单直观，且功能正确)

    # 策略一: 随机选择
    print("\n\n========== 运行随机选择策略 ==========")
    simulator_random = FLSimulator(config)
    results_random = simulator_random.run(selection_strategy='random')
    results_dict['Random Selection'] = results_random

    # 策略二: 资源最优选择
    print("\n\n========== 运行资源最优选择策略 ==========")
    simulator_resource = FLSimulator(config)
    results_resource = simulator_resource.run(selection_strategy='resource_first')
    results_dict['Resource-First Selection'] = results_resource

    # --- 3. 可视化和保存结果 ---
    # 【修改】将output_dir传递给绘图函数
    plot_results(results_dict, output_dir)

    # 【修改】将CSV文件也保存到指定的输出目录
    for strategy, df in results_dict.items():
        csv_path = output_dir / f"results_{strategy.replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"策略 {strategy} 的结果已保存到: {csv_path}")