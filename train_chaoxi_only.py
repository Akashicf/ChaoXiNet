#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from tqdm import tqdm
from mycode.nowcasting.data_provider.npz_loader import *
from mycode.nowcasting.layers.ChaoXiNet.ChaoXiNet import ChaoXiNet
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from mycode.nowcasting.loss_function.loss_evolution import *
from torch import nn

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
use_amp = True


checkpoint_path = '../result/ChaoXiNet'
data_dir = '../data/dataset/yangben_all'
data_dir_h5 = '../data/dataset/yangben_solo_h5'

experiment = 'run1'

num_epochs = 500
lr = 5e-3
batch_size = 2
EvolutionNet = ChaoXiNet( img_size=256, patch_size=4, T=10, in_chans=1, num_classes=1,
                 embed_dim=48*3, depths=[2, 2, 2, 4], num_heads=[4, 4, 8, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.25,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first").cuda()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_dir = os.path.join(checkpoint_path, experiment)
os.makedirs(checkpoint_dir, exist_ok=True)

# 尝试搜集已有 ckpt
ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
ckpts.sort()  # 根据名称排序，最后一个视作最新

start_epoch = 0
global_step = 0  # 用来记录所有batch的迭代次数
train_losses = []
test_losses = []


train_loader, test_loader = get_loaders_h5(data_dir_h5, batch_size=batch_size, num_workers=2)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
optim_evo = torch.optim.AdamW(EvolutionNet.parameters(), lr=lr, betas=(0.5,0.99), weight_decay=0.01)
scheduler = OneCycleLR(
    optim_evo,
    max_lr=lr,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

if len(ckpts) > 0:
    latest_ckpt = ckpts[-1]
    print(f"[INFO] Found checkpoint: {latest_ckpt}, now loading...")
    checkpoint = torch.load(latest_ckpt, map_location=device)

    latest_ckpt = ckpts[-1]
    # 加载模型参数
    EvolutionNet.load_state_dict(checkpoint['EvolutionNet'])
    optim_evo.load_state_dict(checkpoint['optim_evo'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    # 恢复 epoch 及迭代数
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}, global_step={global_step}")

else:
    print("[INFO] No existing checkpoint found, starting from scratch...")
    latest_ckpt = None

def main():
    global train_losses, test_losses, global_step, optim_evo
    for epoch in range(start_epoch, num_epochs):
        # 在 tqdm 上设置 epoch 的提示
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        train_loss = 0.0
        train_count = 0

        t_dataload = time.time()
        for batch_id, test_ims in enumerate(pbar):
            EvolutionNet.train()

            input_frames, target_frames = test_ims
            input_frames = input_frames.to(device)/65
            target_frames = target_frames.to(device)/65

            t_dataload = time.time() - t_dataload
            # ============ 演化网络前向 ============
            t_forward = time.time()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                evo_result = EvolutionNet(input_frames)
                t_forward = time.time() - t_forward
                t_backward = time.time()
                loss_evo = accumulation_loss(
                    pred_final=evo_result,
                    pred_bili=None,
                    real=target_frames,  # [B, T_out, H, W]
                    value_lim = [0, 1]
                )

            scaler.scale(loss_evo).backward()
            scaler.step(optim_evo)
            scaler.update()
            optim_evo.zero_grad()

            t_backward = time.time() - t_backward
            if False:
                print(f'data_load={t_dataload:.3f}, evo_f={t_forward:.3f}, evo_b={t_backward:.3f}')
            t_dataload = time.time()

            # ============ 打印与 tqdm 显示 ============
            pbar.set_postfix({
                'loss_evo': f"{loss_evo.item():.4f}",
            })

            train_loss += loss_evo.item()
            train_count += 1
            global_step += 1  # 全局 iter 计数

            if batch_id == len(train_loader)-1:
                pbar.set_postfix({
                    'loss_evo': f"{train_loss/train_count:.5f}",
                })



        # 将当前batch各项loss以dict形式存下来:
        train_losses.append({
            'global_step': global_step,
            'epoch': epoch,
            'loss_evo': train_loss/train_count,
        })

        # 每 epoch 存一次 ckpt
        if epoch:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step{global_step:06d}.pth")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'train_losses': train_losses,
                'EvolutionNet': EvolutionNet.state_dict(),
                'optim_evo': optim_evo.state_dict(),
                'scheduler': scheduler.state_dict(),
                'test_losses': test_losses,
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint to {ckpt_path}")

        if (global_step + 1) % 10 > -1:
            EvolutionNet.eval()

            # 记录本轮测试的指标
            test_evo_loss = 0.0
            test_count = 0

            # 开始测试循环(不需要梯度)
            with torch.no_grad():
                for batch_id, test_ims in enumerate(test_loader):
                    EvolutionNet.eval()

                    input_frames, target_frames = test_ims
                    input_frames = input_frames.to(device) / 65
                    target_frames = target_frames.to(device) / 65

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        evo_result = EvolutionNet(input_frames)
                        t_forward = time.time() - t_forward
                        t_backward = time.time()
                        loss_evo = accumulation_loss(
                            pred_final=evo_result*65,
                            pred_bili=None,
                            real=target_frames*65,  # [B, T_out, H, W]
                        )

                    # 累加loss
                    test_evo_loss += (loss_evo.item())
                    test_count += 1

            # 计算平均loss
            test_evo_loss /= test_count

            # 保存到 test_losses
            test_losses.append({
                'global_step': global_step,
                'epoch': epoch,
                'loss_evo': test_evo_loss,
            })

            print(
                f"=============================================> Test on step {global_step + 1}: evo_loss={test_evo_loss:.4f}")


if __name__ == '__main__':
    main()