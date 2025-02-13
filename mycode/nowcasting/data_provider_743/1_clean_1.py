import os
import numpy as np
from tqdm import tqdm  # 用于显示进度条
import concurrent.futures
from scipy.ndimage import convolve

def zaboguolv(data, lower_bound, upper_bound):
    """
    使用基于邻域均值填补的方法清洗数据。

    参数：
        data (ndarray): 待清洗的二维数组，形状为 (256, 256)。
        lower_bound (float): 正常值的下边界。
        upper_bound (float): 正常值的上边界。

    返回：
        ndarray: 清洗后的数据。
    """
    # 获取数据的形状
    rows, cols = data.shape

    # 创建结果数组（初始化为原始数据）
    cleaned_data = data.copy()

    # 定义邻域的相对坐标
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1), (0, 1),
                         (1, -1), (1, 0), (1, 1)]

    # 遍历数据中的每个点
    for i in range(rows):
        for j in range(cols):
            # 如果当前点值在正常范围内，跳过处理
            if lower_bound <= data[i, j] <= upper_bound:
                continue

            # 初始化邻域值和权重
            neighbor_values = []
            weights = []

            # 遍历邻域点
            for offset in neighbors_offsets:
                ni, nj = i + offset[0], j + offset[1]
                # 检查邻域点是否在数据范围内
                if 0 <= ni < rows and 0 <= nj < cols:
                    # 检查邻域点是否在正常值范围内
                    if lower_bound <= data[ni, nj] <= upper_bound:
                        neighbor_values.append(data[ni, nj])
                        weights.append(1)

            # 清洗函数处理
            if weights:
                # 使用邻域均值填补
                cleaned_data[i, j] = np.sum(np.array(neighbor_values) * np.array(weights)) / np.sum(weights)
            else:
                # 如果没有有效的邻域点，将值裁剪到正常范围
                cleaned_data[i, j] = np.clip(data[i, j], lower_bound, upper_bound)

    return cleaned_data


def vectorized_zaboguolv(data, lower_bound, upper_bound):
    """
    使用卷积操作实现基于邻域均值填补的数据清洗。
    参数：
        data (ndarray): 待清洗的二维数组，形状为 (H, W)。
        lower_bound (float): 正常值的下边界。
        upper_bound (float): 正常值的上边界。
    返回：
        ndarray: 清洗后的数据。
    """
    # 先创建一份拷贝，便于最后返回结果
    cleaned_data = data.copy()

    # 构造有效值的掩码
    valid_mask = (data >= lower_bound) & (data <= upper_bound)
    valid_float = valid_mask.astype(np.float32)

    # 将无效值置为0，便于后续求和
    data_valid = data * valid_float

    # 定义 3x3 卷积核（全 1 的卷积核）
    kernel = np.ones((3, 3), dtype=np.float32)

    # 计算邻域内有效值的和与计数
    neighbor_sum = convolve(data_valid, kernel, mode='constant', cval=0.0)
    neighbor_count = convolve(valid_float, kernel, mode='constant', cval=0.0)

    # 为避免除 0 错误，先计算邻域均值
    with np.errstate(divide='ignore', invalid='ignore'):
        neighbor_mean = np.where(neighbor_count > 0, neighbor_sum / neighbor_count, 0)

    # 对于不在正常范围内的像素：
    # 如果邻域中有有效点，则使用邻域均值替换；
    # 否则，直接 clip 到正常范围内。
    mask_invalid = ~valid_mask
    # 有效邻域的点
    replace_mask = mask_invalid & (neighbor_count > 0)
    cleaned_data[replace_mask] = neighbor_mean[replace_mask]
    # 邻域中没有有效点的，clip 到范围内
    replace_clip = mask_invalid & (neighbor_count == 0)
    cleaned_data[replace_clip] = np.clip(data[replace_clip], lower_bound, upper_bound)

    return cleaned_data

# 定义各参量和高度对应的上下界
bounds = {
    "dBZ": {
        "1.0km": (0, 70),
        "3.0km": (0, 65),
        "7.0km": (10, 70)
    },
    "ZDR": {
        "1.0km": (-1, 5),
        "3.0km": (-1, 5),
        "7.0km": (-1, 6)
    },
    "KDP": {
        "1.0km": (-0.5, 8),
        "3.0km": (-0.5, 8),
        "7.0km": (-1, 1)
    }
}

# 数据根路径
root_dir = "d:/Desktop/python/NowcastNet/data/dataset/NJU_CPOL_update2308"


def process_file(args):
    """
    单个文件的清洗任务，加载数据、清洗并保存覆盖原文件。

    参数:
        args (tuple): (frame_path, lower_bound, upper_bound)
    """
    frame_path, lower_bound, upper_bound = args
    try:
        data = np.load(frame_path)
        cleaned_data = vectorized_zaboguolv(data, lower_bound, upper_bound)
        np.save(frame_path, cleaned_data)
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
    return frame_path  # 返回处理过的文件路径（便于调试或进度跟踪）


def main():
    # 构造所有待处理文件的任务列表
    tasks = []
    for param in ["dBZ", "ZDR", "KDP"]:
        param_dir = os.path.join(root_dir, param)
        for height in ["1.0km", "3.0km", "7.0km"]:
            height_dir = os.path.join(param_dir, height)
            lower_bound, upper_bound = bounds[param][height]
            for station_folder in os.listdir(height_dir):
                station_path = os.path.join(height_dir, station_folder)
                if os.path.isdir(station_path):
                    for frame_file in os.listdir(station_path):
                        if frame_file.endswith(".npy"):
                            frame_path = os.path.join(station_path, frame_file)
                            tasks.append((frame_path, lower_bound, upper_bound))

    total_files = len(tasks)
    print(f"Total files to process: {total_files}")

    # 使用 ProcessPoolExecutor 并行处理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用 tqdm 包裹 executor.map 以显示进度条
        for _ in tqdm(executor.map(process_file, tasks), total=total_files, desc="数据清洗进度"):
            pass

    print("数据清洗完成！")


if __name__ == '__main__':
    main()
