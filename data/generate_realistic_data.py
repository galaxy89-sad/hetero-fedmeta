# hetero-fedmeta/data/generate_realistic_data.py

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = Path(__file__).parent
FILE_PATHS = {
    "latency": DATA_DIR / "training_latency.json",
    "comm_status": [
        DATA_DIR / "dev_case1.json",
        DATA_DIR / "dev_case2.json",
        DATA_DIR / "mobiperf_tcp_down_2018.json",
        DATA_DIR / "mobiperf_tcp_up_2018.json",
    ],
}
NUM_SAMPLES_TO_GENERATE = 10000
OUTPUT_FILE = DATA_DIR / "normalized_device_profiles_v3.csv"
BINS = {'low': (0.0, 0.33), 'medium': (0.33, 0.66), 'high': (0.66, 1.0)}
RAM_BETA_PARAMS = {'low': {'a': 2, 'b': 5}, 'medium': {'a': 2.5, 'b': 2.5}, 'high': {'a': 5, 'b': 2}}


# --- Helper Functions ---

def load_json_records(filepath):
    """健壮地加载JSON文件，兼容JSON Lines和JSON Array格式"""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # 尝试作为JSON数组解析
            try:
                records = json.loads(content)
            except json.JSONDecodeError:
                # 失败则尝试作为JSON Lines解析
                f.seek(0)
                for line in f:
                    clean_line = line.strip().strip(',[]')
                    if clean_line:
                        try:
                            records.append(json.loads(clean_line))
                        except json.JSONDecodeError:
                            warnings.warn(f"Skipping invalid JSON line in {filepath}: {clean_line[:100]}...")
        # 确保返回的是列表
        return records if isinstance(records, list) else [records]
    except FileNotFoundError:
        warnings.warn(f"File not found: {filepath}. Skipping.")
        return []


def process_power_data(filepath):
    """从时延数据计算原始算力值"""
    print("Processing power data...")
    records = load_json_records(filepath)
    if not records:
        raise ValueError(f"No data found in {filepath}. Cannot proceed.")

    df = pd.DataFrame(records)
    df['Vit Latency'] = pd.to_numeric(df['Vit Latency'], errors='coerce')
    df['ResNet Latency'] = pd.to_numeric(df['ResNet Latency'], errors='coerce')
    df.dropna(subset=['Vit Latency', 'ResNet Latency'], inplace=True)
    df['avg_latency'] = (df['Vit Latency'] + df['ResNet Latency']) / 2
    model_latency = df.groupby('Phone Model')['avg_latency'].mean()
    raw_power = 1 / model_latency
    return pd.DataFrame({'raw_power': raw_power})


def process_communication_data(filepaths):
    """处理所有通信和设备状态数据"""
    print("Processing communication and device status data...")
    all_records = []
    for path in filepaths:
        all_records.extend(load_json_records(path))

    if not all_records:
        raise ValueError("No communication/status data found. Cannot proceed.")

    processed_data = []
    for record in tqdm(all_records, desc="Extracting comm/status records"):
        props = record.get('device_properties')
        if not props:
            continue
        speeds = record.get('tcp_speed_results', [])
        avg_bandwidth = np.mean(speeds) if speeds else np.nan
        raw_rssi = props.get('rssi')
        network_type = props.get('network_type')
        battery = props.get('battery_level')
        charging = props.get('is_battery_charging')
        raw_energy = np.nan
        if battery is not None and charging is not None:
            raw_energy = (1 - battery / 100.0) + (0.2 if not charging else 0)

        processed_data.append(
            {'bandwidth': avg_bandwidth, 'raw_rssi': raw_rssi, 'network_type': network_type, 'energy': raw_energy})

    df = pd.DataFrame(processed_data)
    df.dropna(subset=['bandwidth', 'energy'], inplace=True)

    # 修复了所有FutureWarning的写法
    df['stability'] = df['raw_rssi'].copy()
    df.loc[(df['network_type'] == 'MOBILE') & (df['raw_rssi'] == 99), 'stability'] = np.nan
    df['stability'] = df['stability'].replace({None: np.nan})

    median_rssi = df['stability'].median()
    print(f"Calculated median for RSSI (stability): {median_rssi:.2f}")
    df['stability'] = df['stability'].fillna(median_rssi)

    return df[['bandwidth', 'stability', 'energy']]


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Realistic Data Generation (V3 - with Log Transform) ---")

    # 步骤 1-4
    power_df = process_power_data(FILE_PATHS['latency'])
    comm_df = process_communication_data(FILE_PATHS['comm_status'])
    power_quantiles = power_df['raw_power'].quantile([q for b in BINS.values() for q in b]).to_dict()

    print("Assigning communication data to power-based tiers...")
    comm_indices = np.random.permutation(comm_df.index)
    num_comm_records = len(comm_df)
    proportions = [BINS['low'][1] - BINS['low'][0], BINS['medium'][1] - BINS['medium'][0],
                   BINS['high'][1] - BINS['high'][0]]
    splits = np.round(np.cumsum(proportions) * num_comm_records).astype(int)
    pool_indices_list = np.split(comm_indices, splits[:-1])
    data_pools = {
        'low': comm_df.loc[pool_indices_list[0]],
        'medium': comm_df.loc[pool_indices_list[1]],
        'high': comm_df.loc[pool_indices_list[2]]
    }
    for bin_name, pool_df in data_pools.items():
        print(f"  - Tier '{bin_name}': {len(pool_df)} records.")

    print(f"\nGenerating {NUM_SAMPLES_TO_GENERATE} synthetic samples...")
    synthetic_data = []
    bin_names = list(BINS.keys())
    for _ in tqdm(range(NUM_SAMPLES_TO_GENERATE), desc="Generating samples"):
        chosen_bin = np.random.choice(bin_names)
        power_min = power_quantiles[BINS[chosen_bin][0]]
        power_max = power_quantiles[BINS[chosen_bin][1]]
        power = np.random.uniform(power_min, power_max)
        comm_sample = data_pools[chosen_bin].sample(1).iloc[0]
        bandwidth = comm_sample['bandwidth']
        stability = comm_sample['stability']
        energy = comm_sample['energy']
        ram_params = RAM_BETA_PARAMS[chosen_bin]
        ram = np.random.beta(ram_params['a'], ram_params['b'])
        synthetic_data.append([power, bandwidth, ram, stability, energy])

    columns = ['power', 'bandwidth', 'ram', 'stability', 'energy']
    synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

    # 步骤 5: 对不同列应用最合适的预处理策略
    print("\nApplying tailored pre-processing before normalization...")

    # 对 bandwidth 应用对数变换以修正严重的右偏态
    synthetic_df['bandwidth'] = np.log1p(synthetic_df['bandwidth'])
    print("  - 'bandwidth' transformed using log1p to correct skewness.")

    # 对 stability 仍然使用分位数裁剪
    lower_s = synthetic_df['stability'].quantile(0.01)
    synthetic_df['stability'] = synthetic_df['stability'].clip(lower=lower_s)
    print(f"  - 'stability' clipped at 1st percentile: {lower_s:.4f}")

    # 步骤 6: 进行最终归一化
    print("\nPerforming final Min-Max normalization...")
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(synthetic_df)

    # 转换为DataFrame并保留4位小数
    normalized_df = pd.DataFrame(normalized_data, columns=columns).round(4)

    # 步骤 7: 保存结果并报告
    normalized_df.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Generation Complete! (V3) ---")
    print(f"Successfully generated {len(normalized_df)} samples.")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nSample of the final normalized data:")
    print(normalized_df.head())
    print("\nDescriptive statistics of the final data:")
    print(normalized_df.describe().round(4))