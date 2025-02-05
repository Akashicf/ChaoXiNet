import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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


def get_loaders(data_dir, batch_size=4, shuffle=True, num_workers=16,train_label=200,test_label=None,prefetch_factor=1):
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