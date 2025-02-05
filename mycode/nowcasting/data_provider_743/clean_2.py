import os
import numpy as np
from tqdm import tqdm  # 用于显示进度条
import concurrent.futures


def clean_zdr_with_dbz(zdr_data, dbz_data, zdr_lower=-1, zdr_upper=1, dbz_lower=10, dbz_upper=20):
    """
    使用邻域均值填补 ZDR 数据，但仅在 dBZ 处于 (10, 20) 之间且 ZDR 超出 (-1, 1) 时执行清洗。
    """
    rows, cols = zdr_data.shape
    cleaned_zdr = zdr_data.copy()
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1), (0, 1),
                         (1, -1), (1, 0), (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if dbz_lower < dbz_data[i, j] < dbz_upper:
                if zdr_data[i, j] < zdr_lower or zdr_data[i, j] > zdr_upper:
                    neighbor_values = []
                    for offset in neighbors_offsets:
                        ni, nj = i + offset[0], j + offset[1]
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if zdr_lower <= zdr_data[ni, nj] <= zdr_upper:
                                neighbor_values.append(zdr_data[ni, nj])
                    if neighbor_values:
                        cleaned_zdr[i, j] = np.mean(neighbor_values)
                    else:
                        cleaned_zdr[i, j] = np.clip(zdr_data[i, j], zdr_lower, zdr_upper)
    return cleaned_zdr


def adjust_zdr_with_dbz(zdr_data, dbz_data, zdr_lower=-0.5, zdr_upper=0.5, dbz_lower=10, dbz_upper=20):
    """
    计算满足 dBZ 在 (10, 20) 且 ZDR 在 (-0.5, 0.5) 之间（且 ZDR ≠ zdr[0][0]）的 ZDR 均值，
    然后更新这些 ZDR 值，使其变为 zdr_i = zdr_i - average(zdr)。
    """
    adjusted_zdr = zdr_data.copy()
    exclude_value = zdr_data[0, 0]
    valid_mask = (dbz_lower < dbz_data) & (dbz_data < dbz_upper) & \
                 (zdr_lower < zdr_data) & (zdr_data < zdr_upper) & \
                 (zdr_data != exclude_value)
    valid_zdr_values = zdr_data[valid_mask]
    if valid_zdr_values.size > 0:
        zdr_mean = np.mean(valid_zdr_values)
        adjusted_zdr[valid_mask] -= zdr_mean
    return adjusted_zdr


def process_single_file(zdr_file, dbz_file):
    """
    单个文件的处理任务：加载 ZDR 与 dBZ 数据，
    执行清洗和调整，然后保存覆盖原来的 ZDR 文件。
    """
    try:
        zdr_data = np.load(zdr_file)
        dbz_data = np.load(dbz_file)

        # 清洗操作
        cleaned_zdr = clean_zdr_with_dbz(zdr_data, dbz_data)
        # 调整操作
        adjusted_zdr = adjust_zdr_with_dbz(cleaned_zdr, dbz_data)
        # 保存处理结果
        np.save(zdr_file, adjusted_zdr)
    except Exception as e:
        print(f"Error processing {zdr_file}: {e}")
    return zdr_file


def process_all_data(base_path, altitudes, parameters):
    """
    对指定高度和文件夹中的所有 ZDR 数据执行清洗和调整操作。
    要求 parameters 中必须包含 "dBZ" 和 "ZDR"。
    """
    assert "dBZ" in parameters and "ZDR" in parameters, "必须包含 dBZ 和 ZDR 参数！"
    tasks = []
    for altitude in altitudes:
        zdr_path = os.path.join(base_path, "ZDR", altitude)
        dbz_path = os.path.join(base_path, "dBZ", altitude)
        for dir_name in sorted(os.listdir(zdr_path)):  # 遍历 data_dir_XXX 文件夹
            zdr_dir = os.path.join(zdr_path, dir_name)
            dbz_dir = os.path.join(dbz_path, dir_name)
            if os.path.isdir(zdr_dir) and os.path.isdir(dbz_dir):
                for file_name in sorted(os.listdir(zdr_dir)):  # 遍历 frame_XXX.npy 文件
                    if file_name.endswith(".npy"):
                        zdr_file = os.path.join(zdr_dir, file_name)
                        dbz_file = os.path.join(dbz_dir, file_name)
                        tasks.append((zdr_file, dbz_file))

    total_tasks = len(tasks)
    print(f"总共有 {total_tasks} 个文件需要处理。")

    # 使用 ProcessPoolExecutor 并行处理任务
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用 tqdm 包装并行任务，显示进度条
        futures = [executor.submit(process_single_file, zdr_file, dbz_file) for zdr_file, dbz_file in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="处理进度"):
            pass

    print("所有文件处理完成！")


if __name__ == '__main__':
    # 数据路径及参数
    base_path = "d:/Desktop/python/NowcastNet/data/dataset/NJU_CPOL_update2308"
    altitudes = ["1.0km", "3.0km"]  # 需要处理的高度
    parameters = ["dBZ", "ZDR"]  # dBZ 和 ZDR 数据

    process_all_data(base_path, altitudes, parameters)
