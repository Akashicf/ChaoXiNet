import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

# 数据文件夹路径
folder = r'd:\Desktop\python\NowcastNet\data\dataset\yangben_all'

# 获取所有 npz 文件列表
all_files = [f for f in os.listdir(folder) if f.endswith('.npz')]

# 正则：匹配形如 data_dir_000_sample_0.npz
# 捕获两处数字：1) prefix, 2) sample_idx
pattern = re.compile(r'^data_dir_(\d+)_sample_(\d+)\.npz$')

# 用字典管理每个 prefix 下的所有 sample 文件
# file_dict[prefix] = { sample_idx_0: 文件名, sample_idx_1: 文件名, ... }
file_dict = {}
for fname in all_files:
    match = pattern.match(fname)
    if match:
        prefix_str, sample_str = match.groups()
        prefix = int(prefix_str)
        sample_idx = int(sample_str)
        file_dict.setdefault(prefix, {})[sample_idx] = fname

# 要取的 T 范围
T_range = range(10)  # T=0~9
C1 = 1
C2 = 0

# 遍历每个 prefix
for prefix, sample_dict in file_dict.items():
    # 按照 sample_idx 从小到大排序
    sorted_samples = sorted(sample_dict.keys())

    # 用于存放针对每个 sample 生成的一整列图像（上10张 input + 下10张 output）
    columns_for_concat = []

    for sample_idx in sorted_samples:
        npz_file = os.path.join(folder, sample_dict[sample_idx])
        data = np.load(npz_file)

        # 读取 input 和 output (bfloat16 如果环境不支持，可改 float32 或 float16)
        input_frames = torch.from_numpy(data['input']).bfloat16()   # [T, C1, C2, H, W]
        target_frames = torch.from_numpy(data['output']).bfloat16() # [T, C1, C2, H, W]

        # 把 input 的 T=0~9（C1=1, C2=0）沿纵向堆叠
        input_stacked = []
        for t in T_range:
            # 这里转换成 float32 再 .numpy() 方便可视化
            frame = input_frames[t, C1, C2].float().numpy()
            input_stacked.append(frame)
        input_stacked = np.vstack(input_stacked)  # 在垂直方向(v)拼接 10 张

        # 把 output 的 T=0~9（C1=1, C2=0）沿纵向堆叠
        output_stacked = []
        for t in T_range:
            frame = target_frames[t, C1, C2].float().numpy()
            output_stacked.append(frame)
        output_stacked = np.vstack(output_stacked)  # 垂直方向拼接 10 张

        # 最终这一个 sample 的 20 张图（上10 input，下10 output）再在垂直方向拼接
        # 这样得到一个“大竖条”，高 = 20*H, 宽 = W
        big_column = np.vstack([input_stacked, output_stacked])

        columns_for_concat.append(big_column)

    # 将所有 sample 的竖条在水平方向(h)拼接，从左到右形成最终大图
    if len(columns_for_concat) > 0:
        big_image = np.hstack(columns_for_concat)
    else:
        continue

    # 保存大图
    out_name = f"../img/prefix_{prefix:03d}_all_samples.png"
    plt.imsave(out_name, big_image, cmap='gray')
    print(f"Saved big image for prefix {prefix:03d} as {out_name}")

a = np.load('d:\Desktop\python\\743\data\\NJU_CPOL_update2308\dBZ\\3.0km\data_dir_021\\frame_025.npy')
plt.imshow(a)
plt.colorbar()
plt.show()