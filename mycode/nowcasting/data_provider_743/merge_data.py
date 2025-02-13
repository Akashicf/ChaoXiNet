import os
import glob
import numpy as np
from tqdm import tqdm

data_dir = '../data/dataset/yangben'
all_files = tqdm(sorted(glob.glob(os.path.join(data_dir, '*.npz'))))

inputs = []
outputs = []
for file in all_files:
    data = np.load(file)
    inputs.append(data['input'][:, 1:2, 0:1].transpose(1, 0, 2, 3, 4).copy())
    outputs.append(data['output'][:, 1:2, 0:1].transpose(1, 0, 2, 3, 4).copy())

# 假设各个文件的 'input' 与 'output' 数组可以在第 0 轴上拼接
merged_input = np.concatenate(inputs, axis=0)
merged_output = np.concatenate(outputs, axis=0)

# 保存合并后的数据到一个 npz 文件中
np.savez('merged_data.npz', input=merged_input, output=merged_output)
