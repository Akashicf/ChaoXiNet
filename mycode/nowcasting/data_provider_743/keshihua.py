import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def process_prefix(args):
    """
    处理单个 prefix 下的所有 sample 文件：
      1. 对该 prefix 下的 sample 文件按 sample_idx 排序
      2. 对每个 sample 加载 npz 文件、取出 input 和 output 指定帧，
         拼接成一个长条图像（上10张 input，下10张 output）
      3. 将所有 sample 的长条图像在水平方向拼接成最终大图，并保存
    """
    prefix, sample_dict, folder, T_range, C1, C2 = args
    # 按照 sample_idx 从小到大排序
    sorted_samples = sorted(sample_dict.keys())
    columns_for_concat = []

    for sample_idx in sorted_samples:
        npz_file = os.path.join(folder, sample_dict[sample_idx])
        data = np.load(npz_file)

        # 读取 input 和 output (如果环境不支持 bfloat16，可改为 float32 或 float16)
        input_frames = torch.from_numpy(data['input']).bfloat16()  # [T, C1, C2, H, W]
        target_frames = torch.from_numpy(data['output']).bfloat16()  # [T, C1, C2, H, W]

        # 处理 input：取 T=0~9, C1, C2 指定通道，并转换成 numpy 数组
        input_stacked = []
        for t in T_range:
            frame = input_frames[t, C1, C2].float().numpy()
            input_stacked.append(frame)
        input_stacked = np.vstack(input_stacked)  # 垂直方向拼接 10 张

        # 处理 output
        output_stacked = []
        for t in T_range:
            frame = target_frames[t, C1, C2].float().numpy()
            output_stacked.append(frame)
        output_stacked = np.vstack(output_stacked)  # 垂直方向拼接 10 张

        # 对于一个 sample，将 input（上半部分）和 output（下半部分）垂直拼接
        big_column = np.vstack([input_stacked, output_stacked])
        columns_for_concat.append(big_column)

    # 将所有 sample 的大竖条在水平方向拼接成最终大图
    if columns_for_concat:
        big_image = np.hstack(columns_for_concat)
    else:
        return None

    # 保存大图
    out_name = os.path.join(r"d:\Desktop\python\NowcastNet", "img2", f"prefix_{prefix:03d}_all_samples.png")
    plt.imsave(out_name, big_image, cmap='gray')
    return f"Saved big image for prefix {prefix:03d} as {out_name}"


def main():
    # 数据文件夹路径
    folder = r'd:\Desktop\python\NowcastNet\data\dataset\yangben'

    # 获取所有 npz 文件列表（仅文件名）
    all_files = [f for f in os.listdir(folder) if f.endswith('.npz')]

    # 正则匹配文件名格式： data_dir_000_sample_0.npz
    # 捕获两个数字：1) prefix, 2) sample_idx
    pattern = re.compile(r'^data_dir_(\d+)_sample_(\d+)\.npz$')

    # 将每个 prefix 下的 sample 文件放入字典
    # file_dict[prefix] = { sample_idx: 文件名, ... }
    file_dict = {}
    for fname in all_files:
        match = pattern.match(fname)
        if match:
            prefix_str, sample_str = match.groups()
            prefix = int(prefix_str)
            sample_idx = int(sample_str)
            file_dict.setdefault(prefix, {})[sample_idx] = fname

    # 设置 T 范围及通道选择
    T_range = range(10)  # T=0~9
    C1 = 1
    C2 = 0

    # 构造每个任务的参数元组
    tasks = [(prefix, sample_dict, folder, T_range, C1, C2)
             for prefix, sample_dict in file_dict.items()]

    # 使用多进程并行处理每个 prefix
    with ProcessPoolExecutor() as executor:
        # 使用 tqdm 追踪进度
        for result in tqdm(executor.map(process_prefix, tasks), total=len(tasks)):
            if result:
                print(result)


if __name__ == '__main__':
    main()

# a = np.load('d:\Desktop\python\\743\data\\NJU_CPOL_update2308\dBZ\\3.0km\data_dir_021\\frame_025.npy')
# plt.imshow(a)
# plt.colorbar()
# plt.show()