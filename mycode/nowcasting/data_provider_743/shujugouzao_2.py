import os
import random
import shutil
import numpy as np
import cv2
from tqdm import tqdm  # 进度条
import concurrent.futures

# 原始数据路径
output_dir = "d:/Desktop/python/NowcastNet/data/dataset/yangben"
augmented_dir = "d:/Desktop/python/NowcastNet/data/dataset/yangben_augmented"
os.makedirs(augmented_dir, exist_ok=True)

# -------------------------------
# 筛选极端降水样本部分
# -------------------------------

# 读取所有 npz 文件
npz_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]

extreme_samples = []
for file in tqdm(npz_files, desc="筛选极端降水样本", unit="file"):
    sample_path = os.path.join(output_dir, file)
    data = np.load(sample_path)
    # 取 output 数据，形状 (10, 3, 3, 256, 256)
    output_data = data["output"]
    # 这里假设：1 表示 3.0km，0 表示 ZH（反射率）
    ZH_3km = output_data[:, 1, 0, :, :]
    # 统计每一帧中大于 45 的像素数量
    extreme_areas = np.sum(ZH_3km > 45, axis=(1, 2))
    valid_frames = np.sum(extreme_areas >= 256)
    # 选取至少 3 帧满足条件的样本
    if valid_frames >= 3:
        extreme_samples.append(file)

# 这里直接将所有极端样本作为增强对象（也可以随机采样部分）
samples_to_copy = extreme_samples.copy()
print(f"筛选出 {len(samples_to_copy)} 个极端降水样本。")


# -------------------------------
# 数据增强函数部分
# -------------------------------

# 随机水平或垂直翻转
def random_flip(sample, flip_code):
    flipped_sample = np.copy(sample)
    for t in range(sample.shape[0]):  # 时间步
        for h in range(sample.shape[1]):  # 高度
            for v in range(sample.shape[2]):  # 参量
                flipped_sample[t, h, v, :, :] = cv2.flip(sample[t, h, v, :, :], flip_code)
    return flipped_sample


# 随机旋转 ±5° ~ ±10°
def random_rotate(sample, angle):
    rotated_sample = np.copy(sample)
    for t in range(sample.shape[0]):  # 时间步
        for h in range(sample.shape[1]):  # 高度
            for v in range(sample.shape[2]):  # 参量
                rows, cols = sample.shape[3], sample.shape[4]
                img = sample[t, h, v, :, :].astype(np.float32)
                # 计算旋转矩阵
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                try:
                    rotated_sample[t, h, v, :, :] = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR,
                                                                   borderMode=cv2.BORDER_REFLECT)
                except Exception as e:
                    print(f"错误：无法应用旋转，错误信息: {e}")
                    continue
    return rotated_sample


# 随机平移 ±5~10 像素
def random_translate(sample, tx, ty):
    translated_sample = np.copy(sample)
    for t in range(sample.shape[0]):  # 时间步
        for h in range(sample.shape[1]):  # 高度
            for v in range(sample.shape[2]):  # 参量
                rows, cols = sample.shape[3], sample.shape[4]
                img = sample[t, h, v, :, :].astype(np.float32)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                try:
                    translated_sample[t, h, v, :, :] = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR,
                                                                      borderMode=cv2.BORDER_REFLECT)
                except Exception as e:
                    print(f"错误：无法应用平移，错误信息: {e}")
                    continue
    return translated_sample


# 综合数据增强操作
def augment_data(sample, flip_code, angle, tx, ty):
    augmented_sample = random_flip(sample, flip_code)
    augmented_sample = random_rotate(augmented_sample, angle)
    augmented_sample = random_translate(augmented_sample, tx, ty)
    return augmented_sample


# 生成随机增强配置
def generate_random_settings():
    flip_code = random.choice([0, 1, -1])
    angle_range = random.choice([(-10, -5), (5, 10)])
    angle = random.randint(angle_range[0], angle_range[1])
    tx = random.randint(-10, 10)
    ty = random.randint(-10, 10)
    return flip_code, angle, tx, ty


# -------------------------------
# 并行处理单个文件的增强任务
# -------------------------------

def process_file(file):
    """
    对单个样本文件进行数据增强操作：
    1. 加载 npz 文件中的 input 和 output 数据；
    2. 生成随机增强配置；
    3. 对 input 和 output 均应用相同的增强操作；
    4. 将增强后的数据保存到 augmented_dir 下（文件名保持一致）。
    """
    try:
        sample_path = os.path.join(output_dir, file)
        data = np.load(sample_path)
        input_data = data["input"].astype(np.float16)  # 形状 (10, 3, 3, 256, 256)
        output_data = data["output"].astype(np.float16)  # 形状 (10, 3, 3, 256, 256)

        flip_code, angle, tx, ty = generate_random_settings()
        augmented_input = augment_data(input_data, flip_code, angle, tx, ty)
        augmented_output = augment_data(output_data, flip_code, angle, tx, ty)

        augmented_sample_path = os.path.join(augmented_dir, file)
        np.savez(augmented_sample_path, input=augmented_input, output=augmented_output)
        return file
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")
        return None


# -------------------------------
# 主程序：并行数据增强
# -------------------------------
if __name__ == '__main__':
    # 使用 ProcessPoolExecutor 并行处理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in samples_to_copy]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="数据增强中", unit="file"):
            pass

    print("数据增强完成！")
