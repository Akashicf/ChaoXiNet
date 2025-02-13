import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import h5py


class MyNpzDataset(Dataset):
    """
    一个简单的 Dataset，用于加载包含 'input' 和 'output' 键的 .npz 文件。
    """

    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        data = np.load(npz_path)
        # 读取 input, output
        input_frames = data['input']  # shape 由 npz 内部决定
        target_frames = data['output']  # 同上

        # 转换为 Tensor
        input_frames = torch.from_numpy(input_frames).float()
        target_frames = torch.from_numpy(target_frames).float()

        # 返回 (input, target) 供后续  for batch_id, (input_frames, target_frames) in enumerate(loader):
        return input_frames, target_frames


def get_loaders(data_dir, batch_size=4, shuffle=True, num_workers=16,train_label=200,test_label=None,prefetch_factor=2):
    """
    参数:
      data_dir: 包含所有 .npz 文件的文件夹，例如 '../data/dataset/yangben'
      batch_size, shuffle, num_workers: DataLoader 的常见参数

    返回:
      train_loader, test_loader
    """
    # 1) 获取所有 .npz 文件路径
    all_npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    # 2) 根据文件名中 "xxx" 是否小于200，划分训练、测试集
    #    假设文件名形如:  data_dir_000_sample_0.npz
    #    需要提取其中的 '000' 部分并转为数字
    train_files = []
    test_files = []

    for path in all_npz_files:
        filename = os.path.basename(path)  # e.g. "data_dir_000_sample_0.npz"

        # 方法1: 用 split 切分或正则匹配，此处做一个简易例子:
        # 假设文件名一定包含类似: data_dir_XXX_sample_YYY.npz
        # 则可以 split('_') => ["data","dir","000","sample","0.npz"]（仅示例，视实际情况而定）

        parts = filename.split('_')
        # 最关键是找到第三个部分 "000" => int("000")=0
        # 也可能您需要先过滤 "data","dir" => 具体要看实际文件命名
        # 假设 parts[2]="000"
        number_str = parts[2]  # "000"

        number_val = int(number_str)

        if test_label is not None:
            if number_val < train_label:
                train_files.append(path)
            elif number_val < test_label:
                test_files.append(path)
        else:
            if number_val < train_label:
                train_files.append(path)
            else:
                test_files.append(path)

    # 3) 构造 Dataset
    train_dataset = MyNpzDataset(train_files)
    test_dataset = MyNpzDataset(test_files)

    # 4) 构造 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory = False,
        prefetch_factor = prefetch_factor,
        persistent_workers = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集一般不需要 shuffle
        num_workers=num_workers,
        pin_memory = False,
        prefetch_factor = prefetch_factor,
        persistent_workers = True
    )

    return train_loader, test_loader


class MyNpzDatasetInMemory(Dataset):
    """
    一次性将 file_list 中的所有 .npz 文件读取到内存
    并存放到 self.samples 列表中 (input_frames, target_frames)
    后续 getitem 直接从内存返回，不再读磁盘。
    """

    def __init__(self, file_list):
        super().__init__()
        self.samples = []  # 用于存放所有文件的数据

        for path in file_list:
            data = np.load(path)
            input_frames = torch.from_numpy(data['input']).bfloat16()
            target_frames = torch.from_numpy(data['output']).bfloat16()

            self.samples.append((input_frames, target_frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 直接从 self.samples 列表中根据 idx 取出 (input, target)
        return self.samples[idx]


def get_loaders_in_memory(data_dir, batch_size=4, shuffle=True, num_workers=0,train_label=200,test_label=None):
    # 获取所有文件路径 (与之前类似)
    all_npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    train_files = []
    test_files = []

    for path in all_npz_files:
        filename = os.path.basename(path)
        parts = filename.split('_')
        number_str = parts[2]  # 假设 '000'
        number_val = int(number_str)
        # 自定义切分逻辑
        if test_label is not None:
            if number_val < train_label:
                train_files.append(path)
            elif number_val < test_label:
                test_files.append(path)
        else:
            if number_val < train_label:
                train_files.append(path)
            else:
                test_files.append(path)

    # 使用 "in-memory" 版本的 Dataset
    train_dataset = MyNpzDatasetInMemory(train_files)
    test_dataset = MyNpzDatasetInMemory(test_files)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers>0 对“已经在内存中”的数据帮助不大，反而可能有额外开销
        # 若数据量仍然非常大，也可设置 workers 用于数据增强等
        num_workers=num_workers,
        pin_memory=True,
        # pin_memory_device='cuda:0'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        # pin_memory_device = 'cuda:0'
    )
    return train_loader, test_loader


class MergedNpzDataset(Dataset):
    """
    一个简单的 Dataset，用于加载合并后的 npz 文件，
    假设文件中包含 'input' 和 'output' 键。
    """
    def __init__(self, npz_path):
        super().__init__()
        data = np.load(npz_path)
        self.input_frames = torch.from_numpy(data['input']).float()
        self.target_frames = torch.from_numpy(data['output']).float()

    def __len__(self):
        return self.input_frames.shape[0]

    def __getitem__(self, idx):
        return self.input_frames[idx], self.target_frames[idx]

def get_merged_loader(npz_path, batch_size=4, shuffle=True, num_workers=16, prefetch_factor=2, max_len=10000):
    dataset = MergedNpzDataset(npz_path)
    train_loader = DataLoader(
        dataset[:int(max_len*0.8)],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    test_loader = DataLoader(
        dataset[int(max_len*0.8):],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    return train_loader, test_loader


class MyH5Dataset(Dataset):
    """
    一个简单的 Dataset，用于加载包含 'input' 和 'output' 键的 h5 文件。
    可选地进行数据增强：x、y 轴翻转以及随机旋转 0, 90, 180, 270 度。
    """

    def __init__(self, file_list, augment=False):
        super().__init__()
        self.file_list = file_list
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        h5_path = self.file_list[idx]
        # 每次读取时打开 h5 文件
        with h5py.File(h5_path, 'r') as f:
            input_frames = f['input'][:]  # 例如 shape: (10, 3, 3, 256, 256)
            target_frames = f['output'][:]

        # 转换为 torch tensor（float 类型）
        input_frames = torch.from_numpy(input_frames).float()
        target_frames = torch.from_numpy(target_frames).float()

        # 如果启用数据增强，则对 input 和 target 同步应用增强操作
        if self.augment:
            # 随机沿 x 轴翻转（这里假设最后一维为宽度，即 x 轴）
            if random.random() < 0.5:
                input_frames = torch.flip(input_frames, dims=[-1])
                target_frames = torch.flip(target_frames, dims=[-1])
            # 随机沿 y 轴翻转（倒数第二个维度为高度，即 y 轴）
            if random.random() < 0.5:
                input_frames = torch.flip(input_frames, dims=[-2])
                target_frames = torch.flip(target_frames, dims=[-2])
            # 随机旋转 0, 90, 180, 270 度
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                k = angle // 90  # 每次旋转 90 度
                input_frames = torch.rot90(input_frames, k, dims=[-2, -1])
                target_frames = torch.rot90(target_frames, k, dims=[-2, -1])

        return input_frames, target_frames


def get_loaders_h5(data_dir, batch_size=1, shuffle=True, num_workers=2,
                train_label=200, test_label=None, prefetch_factor=4, augment=True):
    """
    参数:
      data_dir: 包含所有 .h5 文件的文件夹，例如 '../data/dataset/yangben'
      batch_size, shuffle, num_workers: DataLoader 的常见参数
      train_label: 文件名中第三部分（例如 "000"）小于该值的作为训练集（否则为测试集），
                   若 test_label 不为 None，则 train: [0, train_label) test: [train_label, test_label)
      augment: 是否在训练时启用数据增强（测试集一般不启用）

    返回:
      train_loader, test_loader
    """
    # 获取所有 h5 文件路径
    all_h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))

    train_files = []
    test_files = []

    for path in all_h5_files:
        filename = os.path.basename(path)  # e.g. "data_dir_000_sample_0.h5"
        parts = filename.split('_')
        # 假设 parts[2] 为 prefix 数字，如 "000"
        number_str = parts[2]
        number_val = int(number_str)

        if test_label is not None:
            if number_val < train_label:
                train_files.append(path)
            elif number_val < test_label:
                test_files.append(path)
        else:
            if number_val < train_label:
                train_files.append(path)
            else:
                test_files.append(path)

    # 构造 Dataset
    train_dataset = MyH5Dataset(train_files, augment=augment)
    # 测试集通常不做数据增强
    test_dataset = MyH5Dataset(test_files, augment=False)

    # 构造 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    return train_loader, test_loader