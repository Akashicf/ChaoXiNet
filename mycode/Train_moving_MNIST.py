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
        data = np.load(npz_path)  # 根据实际文件结构修改 key
        # 如果是训练集，data 形状可能类似 (60000, 20, 64, 64)
        self.data = torch.tensor(data.astype(np.float32) / 255.0) # 归一化到 [0,1]
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

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        # x, y: [B, 1, 10, 64, 64]
        x, y = x.to(device), y.to(device)

        # 前向
        pred = model(x)  # [B, 1, 10, 64, 64]
        loss = F.mse_loss(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# =========== 准备数据集 ===========
train_npz = '../data/dataset/movingMNIST/mnist_test_seq.npy'
# test_npz  = '../data/dataset/movingMNIST/mnist_1d64_1.npz'
# data = np.load(train_npz)
train_set = MovingMNISTDataset(train_npz, train=True)
# test_set  = MovingMNISTDataset(test_npz,  train=False)

train_loader = DataLoader(train_set, batch_size=24, shuffle=True)
# test_loader  = DataLoader(test_set,  batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========== 初始化模型 ===========
model = QuadrupletSTTSNet(
    image_size=64,
    patch_size=8,
    in_channels=1,  # MovingMNIST 为单通道
    dim=256,        # 可根据显存调整
    depth=6,        # 堆叠 2 层QuadrupletSTTSBlock
    heads=8,
    dim_head=32,
    mlp_dim=1024,
    dropout=0.0,
    T=10
).to(device)

checkpoint_dir = '../result/mnist/run1'

# 尝试搜集已有 ckpt
ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
ckpts.sort()  # 根据名称排序，最后一个视作最新

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
#     # optim_evo.load_state_dict(checkpoint['optim_evo'])
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



optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
train_losses = []


# =========== 训练循环 ===========
num_epochs = 5000  # 演示训练 5 个 epoch
for epoch in range(num_epochs):
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]")
    model.train()
    total_loss = 0
    for index, data_one in enumerate(pbar):
        # print(1)
        # x, y: [B, 1, 10, 64, 64]
        x, y = data_one
        x, y = x.to(device), y.to(device)

        # 前向
        pred = model(x)  # [B, 1, 10, 64, 64]
        loss = F.mse_loss(pred*255, y*255)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
        })

        if index + 1 == len(train_loader):

            train_loss =  total_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            # val_loss   = validate(model, test_loader,  device)

            # print(f"Epoch [{epoch + 1}/{num_epochs}] | Train MSE: {train_loss:.4f}")
            pbar.set_postfix({
                'loss': f"{train_loss:.4f}",
            })

os.mkdir(checkpoint_dir)

ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step{epoch:06d}.pth")
torch.save({
    'epoch': epoch,
    'train_losses': train_losses,
    'EvolutionNet': model.state_dict(),
    'optim_evo': optimizer.state_dict(),
}, ckpt_path)
# 你也可以在此保存模型
# torch.save(model.state_dict(), 'quad_stts_mnist.pth')

# if __name__ == "__main__":
#     main()


index = 4
# 假设 input_tensor, gt_tensor, output_tensor 均为 torch.Tensor，形状为 (B, T, H, W)，B=1
# 若它们已经是 numpy 数组，则可直接使用。
input_tensor = x[index:index+1]
gt_tensor = y[index:index+1]
output_tensor = pred[index:index+1]
# output_tensor = evo_result[index:index+1]

# output_tensor = gen_result

T_in = 10
T_out = 10

# 假设 input_tensor, gt_tensor, output_tensor 均存在，形状为 (B, T, H, W)，B=1
# 这里仅作示例，实际使用时请加载你的 tensor 数据
# 例如：input_tensor = torch.rand(1, T_in, H, W)
# 这里我们直接假设它们已经存在，并转换为 numpy 数组：
input_np = input_tensor[0][0].detach().cpu().numpy()   # shape: (T_in, H, W)
gt_np = gt_tensor[0][0].detach().cpu().numpy()           # shape: (T_out, H, W)
output_np = output_tensor[0][0].detach().cpu().numpy()   # shape: (T_out, H, W)

# 计算 diff（绝对差）
diff_np = np.abs(gt_np - output_np)

# 为了让所有值为0的点显示为白色，我们将使用 masked array，并为对应 colormap 设置 bad 值为白色
def get_masked_array(data):
    # 将所有值为0的点掩码掉
    return np.ma.masked_less(data, -100)

def get_cmap_with_bad(cmap_name):
    # 获取 colormap 副本，并设置bad颜色为白色
    cmap = mpl.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color='white')
    return cmap

# 定义不同列所用的 colormap
cmap_input = get_cmap_with_bad('viridis')
cmap_gt = get_cmap_with_bad('viridis')
cmap_output = get_cmap_with_bad('viridis')
cmap_diff = get_cmap_with_bad('bwr')

# 使用最大的时间步数作为行数
nrows = max(T_in, T_out)
ncols = 4  # 分别为：Input, GT, Output, Diff
vmin, vmax = 0, 1

# 创建图形和子图
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
# 确保 axes 是二维数组
if nrows == 1:
    axes = np.expand_dims(axes, axis=0)
if ncols == 1:
    axes = np.expand_dims(axes, axis=1)

loss_pool_list = []
for i in range(T_out):
    loss_pool =  F.mse_loss(gt_tensor.unsqueeze(2)[:,i:i+1], output_tensor.unsqueeze(2)[:,i:i+1])
    loss_pool_list.append(loss_pool.item())
loss_pool_worst = []
for i in range(T_out):
    loss_pool =  F.mse_loss(gt_tensor.unsqueeze(2)[:,i:i+1], input_tensor.unsqueeze(2)[:,-1:])
    loss_pool_worst.append(loss_pool.item())

# 遍历每一行和每一列绘制图像
for r in range(nrows):
    # 第一列：Input，时间步按降序排列（从上到下 t 从大到小）
    if r < T_in:
        t_idx = T_in - 1 - r  # 降序：上面显示最大的 t
        masked_data = get_masked_array(input_np[t_idx])
        im = axes[r, 0].imshow(masked_data, cmap=cmap_input, vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"Input t={t_idx}")
        fig.colorbar(im, ax=axes[r, 0])
    else:
        axes[r, 0].axis('off')

    # 第二列：GT，时间步按升序排列（从上到下 t 从小到大）
    if r < T_out:
        t_idx = r
        masked_data = get_masked_array(gt_np[t_idx])
        im = axes[r, 1].imshow(masked_data, cmap=cmap_gt, vmin=vmin, vmax=vmax)
        axes[r, 1].set_title(f"GT t={t_idx}")
        fig.colorbar(im, ax=axes[r, 1])
    else:
        axes[r, 1].axis('off')

    # 第三列：Output，时间步按升序排列（从上到下 t 从小到大）
    if r < T_out:
        t_idx = r
        masked_data = get_masked_array(output_np[t_idx])
        im = axes[r, 2].imshow(masked_data, cmap=cmap_output, vmin=vmin, vmax=vmax)
        axes[r, 2].set_title(f"Output t={t_idx}")
        fig.colorbar(im, ax=axes[r, 2])
    else:
        axes[r, 2].axis('off')

    # 第四列：Diff（GT与Output的差值），时间步按升序排列（从上到下 t 从小到大）
    if r < T_out:
        t_idx = r
        masked_data = get_masked_array(diff_np[t_idx])
        im = axes[r, 3].imshow(masked_data, cmap=cmap_diff)
        axes[r, 3].set_title(f"Diff loss={loss_pool_list[t_idx]:.2f}, k={loss_pool_list[t_idx]/(loss_pool_worst[t_idx]+1e-8):.2f}")
        fig.colorbar(im, ax=axes[r, 3])
    else:
        axes[r, 3].axis('off')

# 调整布局并显示图形
plt.tight_layout()
plt.show()

plt.plot(range(len(train_losses)), train_losses)
plt.yscale('log')
plt.xscale('log')
plt.show()