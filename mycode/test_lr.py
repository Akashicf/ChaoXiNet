import numpy as np
import torch
from torch.utils.data import Dataset
from mycode.nowcasting.layers.PredFormer.PredFormer import *
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as mpl
# pip install torchview netron
import torchview
import torchvision
import netron
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
from torch.optim.lr_scheduler import OneCycleLR

# 必须在导入任何依赖 OpenMP 的库之前设置该环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class MovingMNISTDataset(Dataset):
    """
    读取 (N, num_frames, H, W) 的 .npz 文件
    - 训练集: e.g. (60000, 20, 64, 64)
    - 测试集: e.g. (10000, 20, 64, 64)
    默认把前 10 帧作为输入 X，后 10 帧作为预测目标 Y
    """
    def __init__(self, npz_path, train=True):
        super().__init__()
        data = np.load(npz_path)[:,:1]  # 根据实际文件结构修改 key
        # 如果是训练集，data 形状可能类似 (60000, 20, 64, 64)
        self.data = torch.tensor(data.astype(np.float32) / 255.0).cuda() # 归一化到 [0,1]
        self.train = train

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        vid = self.data[:, idx]  # (20, 64, 64)
        # 前 10 帧 (C=1) -> [1, 10, 64, 64], 后 10 帧 -> [1, 10, 64, 64]
        x = vid[:10][None, ...]  # shape [1, 10, 64, 64]
        y = vid[10:][None, ...]  # shape [1, 10, 64, 64]

        # x = torch.from_numpy(x)  # float tensor
        # y = torch.from_numpy(y)
        return x, y



# =========== 准备数据集 ===========
train_npz = '../data/dataset/movingMNIST/mnist_test_seq.npy'
# test_npz  = '../data/dataset/movingMNIST/mnist_1d64_1.npz'
# data = np.load(train_npz)
train_set = MovingMNISTDataset(train_npz, train=True)
# test_set  = MovingMNISTDataset(test_npz,  train=False)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
# test_loader  = DataLoader(test_set,  batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========== 初始化模型 ===========


checkpoint_dir = '../result/mnist/run1'

# 尝试搜集已有 ckpt
ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
ckpts.sort()  # 根据名称排序，最后一个视作最新
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

# if len(ckpts) > 0:
#     latest_ckpt = ckpts[-1]
#     print(f"[INFO] Found checkpoint: {latest_ckpt}, now loading...")
#     checkpoint = torch.load(latest_ckpt, map_location='cuda:0')
#
#     latest_ckpt = ckpts[-1]
#     # 加载模型参数
#     model.load_state_dict(checkpoint['EvolutionNet'])
#
#     # 加载优化器参数
#     optimizer.load_state_dict(checkpoint['optim_evo'])
#     # optim_gen.load_state_dict(checkpoint['optim_gen'])
#     # optim_disc.load_state_dict(checkpoint['optim_disc'])
#
#     # 恢复 epoch 及迭代数
#     start_epoch = checkpoint.get('epoch', 0)
#     global_step = checkpoint.get('global_step', 0)
#     train_losses = checkpoint.get('train_losses', [])
#     test_losses = checkpoint.get('test_losses', [])
#     print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}, global_step={global_step}")
#
# else:
#     print("[INFO] No existing checkpoint found, starting from scratch...")
#     latest_ckpt = None

num_epochs=500



# ====== 自定义模型类 ======

# 假设你已经有 train_loader，以及 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 示例：
# train_loader = ...
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 需要测试的一系列学习率
lrs = [1e-1, 1e-2, 1e-3, 1e-4]

# 用于保存不同 lr 对应的 loss 曲线
all_losses = {}

# 进行训练并记录 loss
for lr in lrs:
    print(f"开始训练，学习率: {lr}")
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model = QuadrupletSTTSNet(
        image_size=64,
        patch_size=8,
        in_channels=1,  # MovingMNIST 为单通道
        dim=256,  # 可根据显存调整
        depth=6,  # 堆叠 2 层QuadrupletSTTSBlock
        heads=8,
        dim_head=32,
        mlp_dim=1024,
        dropout=0.,
        T=10
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.5, 0.999))

    # if len(ckpts) > 0:
    #     latest_ckpt = ckpts[-1]
    #     print(f"[INFO] Found checkpoint: {latest_ckpt}, now loading...")
    #     checkpoint = torch.load(latest_ckpt, map_location='cuda:0')
    #
    #     latest_ckpt = ckpts[-1]
    #     # 加载模型参数
    #     model.load_state_dict(checkpoint['EvolutionNet'])
    #
    #     # 加载优化器参数
    #     optimizer.load_state_dict(checkpoint['optim_evo'])
    #     # optim_gen.load_state_dict(checkpoint['optim_gen'])
    #     # optim_disc.load_state_dict(checkpoint['optim_disc'])
    #
    #     # 恢复 epoch 及迭代数
    #     start_epoch = checkpoint.get('epoch', 0)
    #     global_step = checkpoint.get('global_step', 0)
    #     train_losses = checkpoint.get('train_losses', [])
    #     test_losses = checkpoint.get('test_losses', [])
    #     print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}, global_step={global_step}")



    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    train_losses = []

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        # pbar = tqdm(train_loader, total=len(train_loader), desc=f"LR={lr} | Epoch [{epoch + 1}/{num_epochs}]")
        model.train()
        total_loss = 0.0

        for index, data_one in enumerate(train_loader):
            x, y = data_one
            x, y = x.to(device), y.to(device)

            # 前向
            pred = model(x)  # [B, 1, 10, 64, 64] (仅示例)

            loss = nn.MSELoss()(pred, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_mse = F.mse_loss(pred, y)
            total_loss += loss_mse.item() * x.size(0)

            pbar.set_postfix({'batch_loss': f"{loss_mse.item():.4f}"})

        epoch_loss = total_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
    print(torch.max(pred))
    # plt.imshow(pred.cpu().detach().numpy()[0, 0, 0])
    # plt.show()


    all_losses[lr] = train_losses

# ============= 绘制所有学习率的 loss 曲线 =============
plt.figure(figsize=(10, 6))
for lr in lrs:
    plt.plot(all_losses[lr], label=f"LR={lr}")
plt.xlabel('Epoch')
plt.ylabel('Train MSE Loss')
plt.title('Training Loss Curves for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.yscale('log')
# plt.ylim(0.0407, 0.041)
plt.show()

# all_losses1 = all_losses
# all_losses2 = all_losses
for index, data_one in enumerate(train_loader):
    x, y = data_one
    x, y = x.to(device), y.to(device)

    # 前向
    pred = model(x)

# plt.imshow(pred.cpu().detach().numpy()[0,0,0])
# plt.show()
#
plt.imshow(EvolutionNet.pos_embed.cpu().detach().numpy()[0])
plt.show()
