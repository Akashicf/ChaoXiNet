import os
import time
import numpy as np
import torch
import h5py
import zarr

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原始 npz 文件路径（请根据实际情况调整路径）
base_file = r'd:\Desktop\python\NowcastNet\data\dataset\yangben\data_dir_000_sample_0.npz'

# 存储转换后文件的目录
save_dir = r'd:\Desktop\python\NowcastNet\data\dataset\yangben\converted'
os.makedirs(save_dir, exist_ok=True)

# ---------------------------
# 1. 读取原始 npz 数据
# ---------------------------
data = np.load(base_file)
input_arr = data['input']
output_arr = data['output']

# ---------------------------
# 2. 存储为不同格式
# ---------------------------

# (1) 单独的 npy 文件（分别保存 input 和 output）
npy_input_path = os.path.join(save_dir, 'input.npy')
npy_output_path = os.path.join(save_dir, 'output.npy')
np.save(npy_input_path, input_arr)
np.save(npy_output_path, output_arr)

# (2) 未压缩的 npz 文件
npz_uns_path = os.path.join(save_dir, 'data_uns.npz')
np.savez(npz_uns_path, input=input_arr, output=output_arr)

# (3) 压缩的 npz 文件
npz_comp_path = os.path.join(save_dir, 'data_comp.npz')
np.savez_compressed(npz_comp_path, input=input_arr, output=output_arr)

# (4) HDF5 文件（使用 gzip 压缩）
hdf5_path = os.path.join(save_dir, 'data.h5')
with h5py.File(hdf5_path, 'w') as f:
    f.create_dataset('input', data=input_arr)
    f.create_dataset('output', data=output_arr)

# (5) Zarr 格式（存为 zarr 数组，默认使用 blosc 压缩）
zarr_dir = os.path.join(save_dir, 'zarr_store')
os.makedirs(zarr_dir, exist_ok=True)
# 将 input 和 output 分别保存为 zarr 数组
zarr.save(os.path.join(zarr_dir, 'input.zarr'), input_arr)
zarr.save(os.path.join(zarr_dir, 'output.zarr'), output_arr)


# ---------------------------
# 3. 定义测试加载函数：累计 10 次读取并转换到 CUDA
# ---------------------------

def test_load_npy():
    total_time = 0.0
    for i in range(10):
        start = time.perf_counter()
        in_arr = np.load(npy_input_path)
        out_arr = np.load(npy_output_path)
        in_tensor = torch.from_numpy(in_arr).to(device)
        out_tensor = torch.from_numpy(out_arr).to(device)
        total_time += time.perf_counter() - start
    print(f"npy: 累计10次加载 + to cuda 耗时: {total_time:.6f} s")


def test_load_npz_uns():
    total_time = 0.0
    for i in range(10):
        start = time.perf_counter()
        data_uns = np.load(npz_uns_path)
        in_arr = data_uns['input']
        out_arr = data_uns['output']
        in_tensor = torch.from_numpy(in_arr).to(device)
        out_tensor = torch.from_numpy(out_arr).to(device)
        total_time += time.perf_counter() - start
    print(f"未压缩 npz: 累计10次加载 + to cuda 耗时: {total_time:.6f} s")


def test_load_npz_comp():
    total_time = 0.0
    for i in range(10):
        start = time.perf_counter()
        data_comp = np.load(npz_comp_path)
        in_arr = data_comp['input']
        out_arr = data_comp['output']
        in_tensor = torch.from_numpy(in_arr).to(device)
        out_tensor = torch.from_numpy(out_arr).to(device)
        total_time += time.perf_counter() - start
    print(f"压缩 npz: 累计10次加载 + to cuda 耗时: {total_time:.6f} s")


def test_load_hdf5():
    total_time = 0.0
    for i in range(10):
        start = time.perf_counter()
        with h5py.File(hdf5_path, 'r') as f:
            in_arr = f['input'][:]
            out_arr = f['output'][:]
        in_tensor = torch.from_numpy(in_arr).to(device)
        out_tensor = torch.from_numpy(out_arr).to(device)
        total_time += time.perf_counter() - start
    print(f"HDF5: 累计10次加载 + to cuda 耗时: {total_time:.6f} s")


def test_load_zarr():
    total_time = 0.0
    for i in range(10):
        start = time.perf_counter()
        # 使用 zarr.load 直接加载数组（zarr.load 返回 numpy 数组）
        in_arr = zarr.load(os.path.join(zarr_dir, 'input.zarr'))
        out_arr = zarr.load(os.path.join(zarr_dir, 'output.zarr'))
        in_tensor = torch.from_numpy(in_arr).to(device)
        out_tensor = torch.from_numpy(out_arr).to(device)
        total_time += time.perf_counter() - start
    print(f"Zarr: 累计10次加载 + to cuda 耗时: {total_time:.6f} s")


# ---------------------------
# 4. 运行测试
# ---------------------------
if __name__ == '__main__':
    print("测试不同存储格式的累计加载时间（10次读取）：")
    test_load_npy()
    test_load_npz_uns()
    test_load_npz_comp()
    test_load_hdf5()
    test_load_zarr()
