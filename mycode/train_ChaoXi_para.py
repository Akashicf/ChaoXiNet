#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import random
import argparse

import h5py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

# 如果需要自定义的 ChaoXiNet 和损失函数，请根据实际情况修改下面的 import
# 例如:
# from mycode.nowcasting.layers.ChaoXiNet.ChaoXiNet import ChaoXiNet
# from mycode.nowcasting.loss_function.loss_evolution import accumulation_loss
#
# 这里为了演示，假设您已在相应位置实现好这两个函数 / 类。
from mycode.nowcasting.layers.ChaoXiNet.ChaoXiNet import ChaoXiNet
from mycode.nowcasting.loss_function.loss_evolution import accumulation_loss

# -----------------------------
# 您提供的 Dataset 与 Loader
# -----------------------------
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
            input_frames = f['input'][:]  # 根据您的数据 shape
            target_frames = f['output'][:]

        # 转换为 torch tensor（float 类型）
        input_frames = torch.from_numpy(input_frames).float()
        target_frames = torch.from_numpy(target_frames).float()

        # 如果启用数据增强，则对 input 和 target 同步应用增强操作
        if self.augment:
            # 随机沿 x 轴翻转
            if random.random() < 0.5:
                input_frames = torch.flip(input_frames, dims=[-1])
                target_frames = torch.flip(target_frames, dims=[-1])
            # 随机沿 y 轴翻转
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


def get_loaders_h5(data_dir, batch_size=1, shuffle=True, num_workers=0,
                   train_label=200, test_label=None, prefetch_factor=4, augment=True):
    """
    参数:
      data_dir: 包含所有 .h5 文件的文件夹
      batch_size, shuffle, num_workers: DataLoader 的常见参数
      train_label: 文件名中某部分数字 < train_label 的文件作为训练集
      test_label: 若不为 None，则数字 < test_label 作为测试集
      augment: 是否对训练集进行数据增强

    返回:
      train_loader, test_loader
    """
    # 获取所有 h5 文件路径
    all_h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))

    train_files = []
    test_files = []

    for path in all_h5_files:
        filename = os.path.basename(path)
        parts = filename.split('_')
        # 假设文件名类似：prefix_abc_000.h5
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

    # 构造 Dataset
    train_dataset = MyH5Dataset(train_files, augment=augment)
    test_dataset = MyH5Dataset(test_files, augment=False)

    # 构造 DataLoader (此处未使用 DistributedSampler，若需要可自行添加)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, test_loader

# -----------------------------
# 分布式训练部分
# -----------------------------

use_amp = True
checkpoint_path = '../result/ChaoXiNet'
data_dir_h5 = '../data/dataset/yangben_solo_h5'
experiment = 'run1'

num_epochs = 500
lr = 5e-3
batch_size = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank. Necessary for using the torch.distributed.launch or torchrun")
    args = parser.parse_args()
    return args

def main_worker(local_rank):

    # 1. 初始化进程组
    dist.init_process_group(backend='Gloo')
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2. 创建模型并移动到对应 GPU
    model = ChaoXiNet(
        img_size=256, patch_size=4, T=10, in_chans=1, num_classes=1,
        embed_dim=48*3, depths=[2, 2, 2, 4], num_heads=[4, 4, 8, 8],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.25,
        norm_layer=torch.nn.LayerNorm, ape=True, patch_norm=True,
        use_checkpoint=False, final_upsample="expand_first"
    ).to(device)

    # 3. 用 DDP 包裹模型
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 4. 数据加载器
    train_loader, test_loader = get_loaders_h5(
        data_dir=data_dir_h5,
        batch_size=batch_size,
        shuffle=True,       # 如果使用分布式采样则这里可以置为False
        num_workers=2,
        augment=True
    )

    # 5. 定义优化器、调度器
    optim_evo = torch.optim.AdamW(ddp_model.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=0.01)
    scheduler = OneCycleLR(
        optim_evo,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    # AMP 相关
    scaler = GradScaler(enabled=use_amp)

    # 断点续训相关变量
    checkpoint_dir = os.path.join(checkpoint_path, experiment)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    ckpts.sort()
    start_epoch = 0
    global_step = 0
    train_losses = []
    test_losses = []

    if len(ckpts) > 0:
        # 只在主进程输出信息
        if local_rank == 0:
            print(f"[INFO] Found checkpoint: {ckpts[-1]}, now loading...")

        checkpoint = torch.load(ckpts[-1], map_location=device)
        ddp_model.load_state_dict(checkpoint['EvolutionNet'])
        optim_evo.load_state_dict(checkpoint['optim_evo'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])

        if local_rank == 0:
            print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}, global_step={global_step}")
    else:
        if local_rank == 0:
            print("[INFO] No existing checkpoint found, starting from scratch...")

    # ---------------------------
    # 正式开始训练
    # ---------------------------
    for epoch in range(start_epoch, num_epochs):

        ddp_model.train()
        # 只在主进程上打印进度条
        if local_rank == 0:
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        else:
            pbar = train_loader

        epoch_loss_sum = 0.0
        epoch_count = 0

        for batch_id, (input_frames, target_frames) in enumerate(pbar):
            input_frames = input_frames.to(device) / 65.0
            target_frames = target_frames.to(device) / 65.0

            with autocast(enabled=use_amp):
                evo_result = ddp_model(input_frames)
                loss_evo = accumulation_loss(
                    pred_final=evo_result,
                    pred_bili=None,
                    real=target_frames,  # [B, T_out, H, W]
                    value_lim=[0, 1]
                )

            scaler.scale(loss_evo).backward()
            scaler.step(optim_evo)
            scaler.update()
            optim_evo.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss_sum += loss_evo.item()
            epoch_count += 1
            global_step += 1

            if local_rank == 0:
                pbar.set_postfix({"train_loss": f"{epoch_loss_sum/epoch_count:.4f}"})

        # ========== 只在主进程上进行日志记录和保存 ==========
        if local_rank == 0:
            avg_train_loss = epoch_loss_sum / epoch_count
            train_losses.append({
                'global_step': global_step,
                'epoch': epoch,
                'loss_evo': avg_train_loss,
            })

            # 保存 checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step{global_step:06d}.pth")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'EvolutionNet': ddp_model.module.state_dict(),
                'optim_evo': optim_evo.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint to {ckpt_path}")

        # 所有进程同步，保证主进程写完文件后再进行测试
        dist.barrier()

        # ========== 测试流程 (只在主进程上测试并打印) ==========
        if local_rank == 0:
            ddp_model.eval()
            test_loss_sum = 0.0
            test_count = 0

            with torch.no_grad():
                for (input_frames, target_frames) in test_loader:
                    input_frames = input_frames.to(device) / 65.0
                    target_frames = target_frames.to(device) / 65.0

                    with autocast(enabled=use_amp):
                        evo_result = ddp_model(input_frames)
                        loss_evo = accumulation_loss(
                            pred_final=evo_result * 65.0,
                            pred_bili=None,
                            real=target_frames * 65.0
                        )
                    test_loss_sum += loss_evo.item()
                    test_count += 1

            test_loss_avg = test_loss_sum / test_count if test_count > 0 else 0.0
            test_losses.append({
                'global_step': global_step,
                'epoch': epoch,
                'loss_evo': test_loss_avg,
            })
            print(f"==> Test at step {global_step}, epoch {epoch}: evo_loss={test_loss_avg:.4f}")

        dist.barrier()  # 同步所有进程后再进入下一轮

    # 训练全部结束后，销毁进程组
    dist.destroy_process_group()


def main():
    args = parse_args()
    main_worker(args.local_rank)

if __name__ == '__main__':
    main()
