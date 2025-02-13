import os
import numpy as np
from tqdm import tqdm  # 用于显示进度条
import concurrent.futures

# 设置阈值
I_threshold = 35  # dBZ，强度阈值
A_threshold = 256  # km^2，面积阈值


# 计算面积函数
def calculate_area(data, threshold):
    """
    计算区域内超过强度阈值的面积。
    """
    return np.sum(data >= threshold)


def process_station(data_dir, base_path, output_dir):
    """
    处理单个站点文件夹（例如 data_dir_000），构造样本并保存。
    """
    # 二级菜单
    variables = ["dBZ", "ZDR", "KDP"]
    levels = ["1.0km", "3.0km", "7.0km"]

    # 构造样本路径字典
    sample_paths = {
        level: {var: os.path.join(base_path, var, level, data_dir) for var in variables}
        for level in levels
    }

    # 获取当前站点在 3.0km 层的 dBZ 文件列表，并排序
    station_path = sample_paths["3.0km"]["dBZ"]
    if not os.path.exists(station_path):
        print(f"目录不存在: {station_path}")
        return

    frames = sorted([f for f in os.listdir(station_path) if f.endswith(".npy")])

    # 对每个可能的样本起始时刻（确保有20帧连续数据）
    for t in range(len(frames) - 20):
        # 初始化样本容器
        input_data = []
        output_data = []

        # 使用 3.0km 的 dBZ 数据作为筛选条件
        valid_sample = False
        for i in range(10, 20):
            frame_path = os.path.join(sample_paths["3.0km"]["dBZ"], frames[t + i])
            ZH_3km = np.load(frame_path)
            A_evaluation_t = calculate_area(ZH_3km, I_threshold)
            if A_evaluation_t >= A_threshold:
                valid_sample = True  # 只要有一帧满足条件，就标记为有效
                break

        # 如果样本无效，则跳过
        if not valid_sample:
            continue

        # 构造输入数据（前10帧）
        for i in range(10):
            sample_frame = []
            for level in levels:
                # 对应变量 dBZ, ZDR, KDP
                ZH = np.load(os.path.join(sample_paths[level]["dBZ"], frames[t + i])).astype(np.float16)
                ZDR = np.load(os.path.join(sample_paths[level]["ZDR"], frames[t + i])).astype(np.float16)
                KDP = np.load(os.path.join(sample_paths[level]["KDP"], frames[t + i])).astype(np.float16)
                sample_frame.append((ZH, ZDR, KDP))
            input_data.append(sample_frame)

        # 构造输出数据（后10帧）
        for i in range(10):
            sample_frame = []
            for level in levels:
                ZH = np.load(os.path.join(sample_paths[level]["dBZ"], frames[t + 10 + i])).astype(np.float16)
                ZDR = np.load(os.path.join(sample_paths[level]["ZDR"], frames[t + 10 + i])).astype(np.float16)
                KDP = np.load(os.path.join(sample_paths[level]["KDP"], frames[t + 10 + i])).astype(np.float16)
                sample_frame.append((ZH, ZDR, KDP))
            output_data.append(sample_frame)

        # 转换为 numpy 数组并存储为 .npz 压缩文件
        # 数据形状：(10, 3, 256, 256, 3)
        input_data = np.array(input_data, dtype=np.float16)
        output_data = np.array(output_data, dtype=np.float16)

        sample_file = os.path.join(output_dir, f"{data_dir}_sample_{t}.npz")
        np.savez_compressed(sample_file, input=input_data, output=output_data)
        print(f"Sample {sample_file} saved.")


def construct_samples_parallel(base_path, num_frames, output_dir):
    """
    根据指定路径构造样本（并行处理），并进行欠采样处理。
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历所有站点文件夹，例如 data_dir_000 到 data_dir_257
    station_dirs = [f"data_dir_{i:03d}" for i in range(258)]

    # 使用 ProcessPoolExecutor 并行处理各站点
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 构造任务列表，每个任务对应一个站点
        futures = []
        for data_dir in station_dirs:
            futures.append(executor.submit(process_station, data_dir, base_path, output_dir))

        # 使用 tqdm 跟踪进度
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="样本构造进度"):
            pass


# 主程序入口
if __name__ == '__main__':
    base_path = "d:/Desktop/python/NowcastNet/data/dataset/NJU_CPOL_update2308"  # 根路径
    output_dir = "d:/Desktop/python/NowcastNet/data/dataset/yangben"  # 输出路径
    num_frames = 100  # 每个文件夹中的帧数（根据实际情况调整，可用于后续扩展）

    construct_samples_parallel(base_path, num_frames, output_dir)
