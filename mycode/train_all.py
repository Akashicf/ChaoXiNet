#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import cv2
import numpy as np
import torch
import time
import sys
import glob
from tqdm import tqdm
import matplotlib as mpl
# ===== 如果你需要下面的可视化功能，请确保已安装 =====
# pip install torchview netron
import torchview
import torchvision
import netron

# =========== 模型相关的依赖 =============
from mycode.nowcasting.data_provider import datasets_factory
from mycode.nowcasting.data_provider.npz_loader import *
from mycode.nowcasting.layers.utils import warp, make_grid

from mycode.nowcasting.models.nowcastnet import Net  # 示例：Net是NowcastNet的网络类
from mycode.nowcasting.models.temporal_discriminator import Temporal_Discriminator
from mycode.nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from mycode.nowcasting.layers.evolution.evolution_network import Evolution_Network
from mycode.nowcasting.layers.generation.noise_projector import Noise_Projector

from mycode.nowcasting.loss_function.loss_evolution import *
from mycode.nowcasting.loss_function.loss_discriminator import *
from mycode.nowcasting.loss_function.loss_generation import *
# =========== 一些评估相关的依赖 (如果需要的话) ===========
import mycode.nowcasting.evaluator as evaluator

# 打开cudnn 加速
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True




################################################################################
#                            模型封装类 (合并自 model_factory.py)
################################################################################
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        # 根据名称选择网络，这里假设只有 NowcastNet 一种
        networks_map = {
            'NowcastNet': Net,
        }
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            # 实例化网络
            self.network = Network(configs).to(configs.device)
            # 加载预训练参数
            self.test_load()
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

    def test_load(self):
        if not self.configs.pretrained_model:
            raise ValueError("请指定 --pretrained_model 参数，否则无法加载模型权重！")
        stats = torch.load(self.configs.pretrained_model, map_location=self.configs.device)
        self.network.load_state_dict(stats)

    def test(self, frames):
        """对输入数据进行预测推理"""
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        self.network.eval()
        with torch.no_grad():
            next_frames = self.network(frames_tensor)
            print(frames_tensor.shape)

        return next_frames.detach().cpu().numpy()


################################################################################
#               测试函数包装 (合并自 run.py 中的 test_wrapper_pytorch_loader)
################################################################################
def test_wrapper_pytorch_loader(model, args):
    # 读取数据
    batch_size_test = args.batch_size
    test_input_handle = datasets_factory.data_provider(args)
    args.batch_size = batch_size_test

    # 使用 evaluator 中的测试函数 (如果你有自己的测试逻辑可以自行修改)
    evaluator.test_pytorch_loader(model, test_input_handle, args, 'test_result')


################################################################################
#                        可视化方法 (torchview & Netron)
################################################################################
def visualize_model(model, args):
    """
    1) torchview 直接绘制网络结构图
    2) 导出 ONNX 后用 netron 可视化
    """
    # 做一个假输入，形状要和你的网络期望的输入相匹配
    # 例如 NowcastNet 可能是 (batch, in_seq, channels, height, width)
    dummy_input = torch.randn(
        args.batch_size,
        9,
        args.img_height,
        args.img_width,
        args.img_ch
    ).to(args.device)

    print("\n>>> [1] 使用 torchview 可视化网络结构图 ...")
    graph = torchview.draw_graph(model.network, input_size=dummy_input.shape, expand_nested=False, save_graph=True,
                         filename="torchview_decoder", roll=False,hide_inner_tensors=True,graph_dir="UD")
    # 保存为 PDF 或 SVG 等格式
    graph.visual_graph
    print("已保存 torchview 可视化结构图到 model_torchview.pdf")

    # print("\n>>> [2] 使用 netron 可视化网络结构 (ONNX) ...")
    # # 先导出 ONNX
    # onnx_path = "nowcastnet.onnx"
    # torch.onnx.export(
    #     model.network,
    #     dummy_input,
    #     onnx_path,
    #     input_names=['input'],
    #     output_names=['output'],
    #     opset_version=16 # 根据需要可指定更高版本
    # )
    # print(f"已导出模型为 {onnx_path} ... 正在启动 netron 可视化服务 (默认端口9999)")
    #
    # # 启动 netron
    # netron.start(onnx_path, port=9999, browse=True)
    # 注意：netron 会阻塞在此，直到你手动关闭可视化窗口或 ctrl+c


################################################################################
#                             主函数 (合并自 run.py)
################################################################################

if 'pydevconsole.py' in sys.argv[0]:
    sys.argv = sys.argv[:1]

parser = argparse.ArgumentParser(description='NowcastNet Inference & Visualization')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--worker', type=int, default=16)
parser.add_argument('--cpu_worker', type=int, default=16)
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--dataset_path', type=str, default='../data/dataset/mrms/figure')
parser.add_argument('--dataset_path_train', type=str, default='../data/dataset/mrms/us_eval.split/MRMS_Final_Test_Patch')
parser.add_argument('--model_name', type=str, default='NowcastNet')
parser.add_argument('--pretrained_model', type=str, default='data/checkpoints/mrms_model.ckpt')
parser.add_argument('--gen_frm_dir', type=str, default='results/us/')
parser.add_argument('--case_type', type=str, default='normal')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--img_width', type=int, default=256)
parser.add_argument('--img_ch', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_save_samples', type=int, default=1000)
parser.add_argument('--ngf', type=int, default=32)

# 你可以加一个可视化开关
parser.add_argument('--visualize', action='store_true',
                    help='是否使用 torchview 和 netron 进行模型结构可视化')

args, unknown = parser.parse_known_args()


# 其他派生参数
args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.pred_length = args.total_length - args.input_length

args.ic_feature = args.ngf * 10

args.pool_loss_k = 1
args.use_num = 80000
args.checkpoint_path = '../result'
args.experiment = 'run17'
# 创建输出文件夹（若已存在则删除重建）
if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print('>>> 初始化并加载模型 ...')
# model = Model(args)

# network = Net(args).to(args.device)
# stats = torch.load('../data/checkpoints/mrms_model.ckpt')
# network.load_state_dict(stats)
#
# EvolutionNet = network.evo_net
# GenerativeEncoder = network.gen_enc
# GenerativeDecoder = network.gen_dec
# NoiseProjector = network.proj

# network = None
EvolutionNet = Evolution_Network(args.input_length*args.input_channel, args.pred_length, base_c=32).to(args.device)
GenerativeEncoder = Generative_Encoder(args.total_length, base_c=args.ngf).to(args.device)
GenerativeDecoder = Generative_Decoder(args).to(args.device)
NoiseProjector = Noise_Projector(args.ngf, args).to(args.device)
TemporalDiscriminator = Temporal_Discriminator(args).to(args.device)




optim_evo = torch.optim.Adam(EvolutionNet.parameters(), lr=10e-4, betas=(0.5,0.999))
optim_gen = torch.optim.Adam(
    list(GenerativeEncoder.parameters())
    + list(GenerativeDecoder.parameters())
    + list(NoiseProjector.parameters()),
    lr=3e-4, betas=(0.5,0.999)
)
optim_disc = torch.optim.Adam(TemporalDiscriminator.parameters(), lr=3e-4, betas=(0.5,0.999))


# >>> 新增: 创建 / 加载 checkpoint 逻辑 <<<
checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment)
os.makedirs(checkpoint_dir, exist_ok=True)

# 尝试搜集已有 ckpt
ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
ckpts.sort()  # 根据名称排序，最后一个视作最新

start_epoch = 0
global_step = 0  # 用来记录所有batch的迭代次数
train_losses = []
test_losses = []

if len(ckpts) > 0:
    latest_ckpt = ckpts[-1]
    print(f"[INFO] Found checkpoint: {latest_ckpt}, now loading...")
    checkpoint = torch.load(latest_ckpt, map_location=args.device)

    # 加载模型参数
    EvolutionNet.load_state_dict(checkpoint['EvolutionNet'])
    GenerativeEncoder.load_state_dict(checkpoint['GenerativeEncoder'])
    GenerativeDecoder.load_state_dict(checkpoint['GenerativeDecoder'])
    NoiseProjector.load_state_dict(checkpoint['NoiseProjector'])
    TemporalDiscriminator.load_state_dict(checkpoint['TemporalDiscriminator'])

    # 加载优化器参数
    optim_evo.load_state_dict(checkpoint['optim_evo'])
    optim_gen.load_state_dict(checkpoint['optim_gen'])
    optim_disc.load_state_dict(checkpoint['optim_disc'])

    # 恢复 epoch 及迭代数
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    train_losses = checkpoint.get('train_losses', [])
    print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}, global_step={global_step}")

else:
    print("[INFO] No existing checkpoint found, starting from scratch...")
    latest_ckpt = None


# d = np.load('d:/Desktop/python/NowcastNet/data/dataset/yangben_augmented/data_dir_000_sample_0.npy')
# torch.from_numpy(d['input']).bfloat16()

batch_size_test = 14
args.batch_size = batch_size_test
# train_loader, test_loader = datasets_factory.data_provider(args)
data_dir = '../data/dataset/yangben'
train_loader, test_loader = get_loaders(data_dir, batch_size=args.batch_size, num_workers=2)

# ckpt2 = torch.load('../data/checkpoints/mrms_model.ckpt', map_location=args.device)
# for x in checkpoint.keys():
#     print(x)
#
# checkpoint['EvolutionNet']
# for x in checkpoint['optim_evo'].keys():
#     print(x)

num_epochs = 2000  # 例如您想训练多少个 epoch
test_interval = 100

# pbar = tqdm(range(num_epochs))
Train = True
# loader = datasets_factory.data_provider(args)  # 可迭代对象

for epoch in range(num_epochs):
    # 在 tqdm 上设置 epoch 的提示

    if Train:
        loader = train_loader
    else:
        loader = test_loader
    if Train:
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]")
    else:
        pbar = tqdm(test_loader, total=len(test_loader), desc="Test")

    t_dataload = time.time()
    for batch_id, test_ims in enumerate(loader):

        EvolutionNet.train()
        GenerativeEncoder.train()
        GenerativeDecoder.train()
        NoiseProjector.train()
        TemporalDiscriminator.train()

        if args.input_channel == 2:
            test_ims = test_ims['radar_frames'].numpy()
            test_ims = torch.FloatTensor(test_ims).to(args.device)

            # (1) 转换形状: B,T,H,W,C
            test_ims = test_ims[:, :, :, :, :1]
            frames = test_ims.permute(0, 1, 4, 2, 3)  # => [B,T,C,H,W]
            batch = frames.shape[0]
            height = frames.shape[3]
            width = frames.shape[4]

            # (2) 拆分输入/目标
            input_frames = frames[:, :args.input_length]           # [B,T_in,C,H,W]
            target_frames = frames[:, args.input_length:, 0]       # [B,T_out,H,W], 只拿通道0
            input_frames = input_frames.reshape(batch, args.input_length, height, width)
            last_frames = test_ims[:, (args.input_length - 1):args.input_length, :, :, 0]

        else:
            input_frames, target_frames = test_ims
            batch, T, C1, C2, height, width = input_frames.shape
            last_frames = input_frames[:, -1:, 1, 0].to(args.device)
            input_frames = input_frames[:, :, 1, 0].reshape(batch, -1, height, width).to(args.device)
            target_frames = target_frames[:, :, 1, 0].to(args.device)

        t_dataload = time.time() - t_dataload
            # ============ 演化网络前向 + 逐帧梯度截断 ============
        t_forward = time.time()
        intensity, motion = EvolutionNet(input_frames)
        motion_ = motion.reshape(batch, args.gen_oc, 2, height, width)
        intensity_ = intensity.reshape(batch, args.pred_length, 1, height, width)

        series = []
        series_bili = []

        sample_tensor = torch.zeros(1, 1, args.img_height, args.img_width).to(args.device)
        grid = make_grid(sample_tensor)
        grid = grid.repeat(batch, 1, 1, 1)

        t_forward = time.time() - t_forward
        t_evo = time.time()

        # EvolutionNet.inc.double_conv[0].weight.grad
        # 多步演化, 每帧截断梯度
        for i in range(args.pred_length):
            x_t = last_frames.detach()

            x_t_dot_bili = warp(x_t, motion_[:, i], grid, mode="bilinear", padding_mode="border")
            x_t_dot = warp(x_t, motion_[:, i], grid, mode="nearest", padding_mode="border")
            x_t_dot_dot = x_t_dot.detach() + intensity_[:, i]
            # last_frames_ = last_frames_

            last_frames = x_t_dot_dot
            series.append(x_t_dot_dot)
            series_bili.append(x_t_dot_bili)

        evo_result = torch.cat(series, dim=1)
        evo_result_bili = torch.cat(series_bili, dim=1)

        t_evo = time.time() - t_evo
        t_backward = time.time()

        # ============ 演化网络损失及更新 ============
        loss_motion = motion_reg(motion_, target_frames)
        loss_accum = accumulation_loss(
            pred_final=evo_result,
            pred_bili=evo_result_bili,
            real=target_frames,  # [B, T_out, H, W]
        )
        loss_evo = loss_accum + 0.01 * loss_motion
        if Train:
            optim_evo.zero_grad()
            loss_evo.backward()
            optim_evo.step()

        gen_result = evo_result
        t_backward = time.time() - t_backward


        t_gen_forward = time.time()
        # detach 演化输出, 避免对抗梯度回传
        evo_result_detach = evo_result.detach() / 65

        # ============ 生成网络 + 判别器 ============

        # 1) 生成器前向
        evo_feature = GenerativeEncoder(torch.cat([input_frames, evo_result_detach], dim=1))

        gen_result_list = []
        dis_result_pre_list = []
        for _ in range(args.pool_loss_k):
            noise = torch.randn(batch, args.ngf, height // 32, width // 32).to(args.device)
            noise_feature = NoiseProjector(noise)
            noise_feature = noise_feature.reshape(
                batch, -1, 4, 4, 8, 8
            ).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

            feature = torch.cat([evo_feature, noise_feature], dim=1)
            gen_result = GenerativeDecoder(feature, evo_result_detach)
            # shape => [B, T_out, H, W], 视实际而定

            gen_result = gen_result.unsqueeze(2)  # => [B,1,H,W] => or [B,T=1,H,W]
            gen_result_list.append(gen_result)

            # 判别器对 fake
            dis_result_pre = TemporalDiscriminator(gen_result, input_frames.unsqueeze(2))
            dis_result_pre_list.append(dis_result_pre)

        t_gen_forward = time.time() - t_gen_forward
        t_gen_backward = time.time()
        # ============ (a) 生成器更新 ============
        loss_adv = adversarial_loss(dis_result_pre_list)  # 生成器骗判别器
        loss_pool = pool_regularization(target_frames.unsqueeze(2), gen_result_list)
        loss_generative = 6 * loss_adv + 20 * loss_pool

        if Train:
            optim_gen.zero_grad()
            loss_generative.backward()
            optim_gen.step()

        t_gen_backward = time.time() - t_gen_backward
        t_dis_forward = time.time()

        # 判别器对 real
        dis_result_GT = TemporalDiscriminator(target_frames.unsqueeze(2), input_frames.unsqueeze(2))
        dis_result_pre = TemporalDiscriminator(gen_result.detach(), input_frames.unsqueeze(2))

        t_dis_forward = time.time() - t_dis_forward
        t_dis_backward = time.time()
        # ============ (b) 判别器更新 ============
        loss_disc = discriminator_loss(dis_result_GT, dis_result_pre)
        if Train:
            optim_disc.zero_grad()
            loss_disc.backward()
            optim_disc.step()

        t_dis_backward = time.time() - t_dis_backward

        # 打印步骤时间
        print(f'data_load={t_dataload:.2f}, evo_f={t_forward:.2f}, evo_phy={t_evo:.2f}, evo_b={t_backward:.2f}, gen_f={t_gen_forward:.2f}, gen_b={t_gen_backward:.2f}, dis_f={t_dis_forward:.2f}, dis_b={t_dis_backward:.2f}')
        t_dataload = time.time()


        # ============ 打印与 tqdm 显示 ============
        # pbar.set_postfix({
        #     'loss_evo': f"{loss_evo.item():.4f}",
        #     'acc': f"{loss_accum.item():.4f}",
        #     'mot': f"{loss_motion.item():.4f}",
        #     'loss_disc': f"{loss_disc.item():.4f}",
        #     'loss_gen': f"{loss_generative.item():.4f}",
        #     'adv': f"{loss_adv.item():.4f}",
        #     'pool': f"{loss_pool.item():.4f}",
        # })




        # >>> 新增: 记录与保存逻辑 <<<
        global_step += 1  # 全局 iter 计数

        # 将当前batch各项loss以dict形式存下来:
        train_losses.append({
            'global_step': global_step,
            'epoch': epoch,
            'loss_evo': loss_evo.item(),
            'loss_accum': loss_accum.item(),
            'loss_motion': loss_motion.item(),
            'loss_disc': loss_disc.item(),
            'loss_gen': loss_generative.item(),
            'loss_adv': loss_adv.item(),
            'loss_pool': loss_pool.item(),
        })

    # 每 100 iter 存一次 ckpt
    if epoch:
        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step{global_step:06d}.pth")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'train_losses': train_losses,
            'EvolutionNet': EvolutionNet.state_dict(),
            'GenerativeEncoder': GenerativeEncoder.state_dict(),
            'GenerativeDecoder': GenerativeDecoder.state_dict(),
            'NoiseProjector': NoiseProjector.state_dict(),
            'TemporalDiscriminator': TemporalDiscriminator.state_dict(),
            'optim_evo': optim_evo.state_dict(),
            'optim_gen': optim_gen.state_dict(),
            'optim_disc': optim_disc.state_dict(),
        }, ckpt_path)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")

    if (global_step + 1) % test_interval > -1:
        # 切换到 eval 模式(如果模型中有BN/Dropout等)
        EvolutionNet.eval()
        GenerativeEncoder.eval()
        GenerativeDecoder.eval()
        NoiseProjector.eval()
        TemporalDiscriminator.eval()

        # 记录本轮测试的指标
        test_evo_loss = 0.0
        test_accum_loss = 0.0
        test_motion_loss = 0.0
        test_disc_loss = 0.0
        test_gen_loss = 0.0
        test_adv_loss = 0.0
        test_pool_loss = 0.0
        test_count = 0

        # 开始测试循环(不需要梯度)
        with torch.no_grad():
            # pbar_test = tqdm(test_loader, total=len(test_loader), desc=f"Test Epoch [{epoch + 1}/{num_epochs}]")
            for batch_id, test_ims in enumerate(test_loader):
                EvolutionNet.eval()
                GenerativeEncoder.eval()
                GenerativeDecoder.eval()
                NoiseProjector.eval()
                TemporalDiscriminator.eval()

                if args.input_channel == 2:
                    test_ims = test_ims['radar_frames'].numpy()
                    test_ims = torch.FloatTensor(test_ims).to(args.device)

                    # (1) 转换形状: B,T,H,W,C
                    test_ims = test_ims[:, :, :, :, :1]
                    frames = test_ims.permute(0, 1, 4, 2, 3)  # => [B,T,C,H,W]
                    batch = frames.shape[0]
                    height = frames.shape[3]
                    width = frames.shape[4]

                    # (2) 拆分输入/目标
                    input_frames = frames[:, :args.input_length]  # [B,T_in,C,H,W]
                    target_frames = frames[:, args.input_length:, 0]  # [B,T_out,H,W], 只拿通道0
                    input_frames = input_frames.reshape(batch, args.input_length, height, width)
                    last_frames = test_ims[:, (args.input_length - 1):args.input_length, :, :, 0]

                else:
                    input_frames, target_frames = test_ims
                    batch, T, C1, C2, height, width = input_frames.shape
                    last_frames = input_frames[:, -1:, 1, 0].to(args.device)
                    input_frames = input_frames[:, :, 1, 0].reshape(batch, -1, height, width).to(args.device)
                    target_frames = target_frames[:, :, 1, 0].to(args.device)

                # ============ 演化网络前向 + 逐帧梯度截断 ============
                intensity, motion = EvolutionNet(input_frames)
                motion_ = motion.reshape(batch, args.gen_oc, 2, height, width)
                intensity_ = intensity.reshape(batch, args.pred_length, 1, height, width)

                series = []
                series_bili = []
                # last_frames = test_ims[:, (args.input_length - 1):args.input_length, :, :, 0]

                sample_tensor = torch.zeros(1, 1, args.img_height, args.img_width).to(args.device)
                grid = make_grid(sample_tensor)
                grid = grid.repeat(batch, 1, 1, 1)

                # 多步演化, 每帧截断梯度
                for i in range(args.pred_length):
                    last_frames = last_frames.detach()

                    last_frames_bili = warp(last_frames, motion_[:, i], grid, mode="bilinear",
                                            padding_mode="border")
                    last_frames_ = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
                    last_frames_ = last_frames_ + intensity_[:, i]
                    last_frames_ = last_frames_.detach()

                    last_frames = last_frames_
                    series.append(last_frames)
                    series_bili.append(last_frames_bili)

                evo_result = torch.cat(series, dim=1)
                evo_result_bili = torch.cat(series_bili, dim=1)

                # ============ 演化网络损失及更新 ============
                loss_motion = motion_reg(motion_, target_frames)
                loss_accum = accumulation_loss(
                    pred_final=evo_result,
                    pred_bili=evo_result_bili,
                    real=target_frames,  # [B, T_out, H, W]
                )
                loss_evo = loss_accum + 0.01 * loss_motion

                # detach 演化输出, 避免对抗梯度回传
                evo_result_detach = evo_result.detach() / 65

                # ============ 生成网络 + 判别器 ============

                # 1) 生成器前向
                evo_feature = GenerativeEncoder(torch.cat([input_frames, evo_result_detach], dim=1))

                gen_result_list = []
                dis_result_pre_list = []
                for _ in range(args.pool_loss_k):
                    noise = torch.randn(batch, args.ngf, height // 32, width // 32).to(args.device)
                    noise_feature = NoiseProjector(noise)
                    noise_feature = noise_feature.reshape(
                        batch, -1, 4, 4, 8, 8
                    ).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

                    feature = torch.cat([evo_feature, noise_feature], dim=1)
                    gen_result = GenerativeDecoder(feature, evo_result_detach)
                    # shape => [B, T_out, H, W], 视实际而定

                    gen_result = gen_result.unsqueeze(2)  # => [B,1,H,W] => or [B,T=1,H,W]
                    gen_result_list.append(gen_result)

                    # 判别器对 fake
                    dis_result_pre = TemporalDiscriminator(gen_result, input_frames.unsqueeze(2))
                    dis_result_pre_list.append(dis_result_pre)

                # ============ (a) 生成器更新 ============
                loss_adv = adversarial_loss(dis_result_pre_list)  # 生成器骗判别器
                loss_pool = pool_regularization(target_frames.unsqueeze(2), gen_result_list)
                loss_generative = 6 * loss_adv + 20 * loss_pool


                # 判别器对 real
                dis_result_GT = TemporalDiscriminator(target_frames.unsqueeze(2), input_frames.unsqueeze(2))
                dis_result_pre = TemporalDiscriminator(gen_result.detach(), input_frames.unsqueeze(2))
                # ============ (b) 判别器更新 ============
                loss_disc = discriminator_loss(dis_result_GT, dis_result_pre)

                # 累加loss
                test_evo_loss += (loss_evo.item())
                test_accum_loss += (loss_accum.item())
                test_motion_loss += (loss_motion.item())
                test_disc_loss += (loss_disc.item())
                test_gen_loss += (loss_generative.item())
                test_adv_loss += (loss_adv.item())
                test_pool_loss += (loss_pool.item())
                test_count += 1

        # 计算平均loss
        test_evo_loss /= test_count
        test_accum_loss /= test_count
        test_motion_loss /= test_count
        test_disc_loss /= test_count
        test_gen_loss /= test_count
        test_adv_loss /= test_count
        test_pool_loss /= test_count

        # 保存到 test_losses
        test_losses.append({
            'global_step': global_step,
            'epoch': epoch,
            'loss_evo': test_evo_loss,
            'loss_accum': test_accum_loss,
            'loss_motion': test_motion_loss,
            'loss_disc': test_disc_loss,
            'loss_gen': test_gen_loss,
            'loss_adv': test_adv_loss,
            'loss_pool': test_pool_loss,
        })

        print(
            f"====> Test on step {global_step + 1}: evo_loss={test_evo_loss:.4f}, acc={test_accum_loss:.4f}, mot={test_motion_loss:.4f}, disc_loss={test_disc_loss:.4f}, gen_loss={test_gen_loss:.4f}, adv={test_adv_loss:.4f}, pool={test_pool_loss:.4f}")

        if not Train:
            break
    if not Train:
        break






# print('>>> 开始测试推理 ...')
# test_wrapper_pytorch_loader(model, args)
# print('>>> 推理完成! 结果已保存在 "{}" 下'.format(args.gen_frm_dir))
#
# # 如果指定可视化，就绘制网络结构
# if args.visualize:
#     print('\n>>> 准备进行模型可视化 ...')
#     visualize_model(model, args)


import torch
import numpy as np
import matplotlib.pyplot as plt

def stack_images_vertically(tensor_np, order='asc'):
    """
    将 shape = (T, H, W) 的 numpy 数组中的每一帧（二维图像）沿垂直方向拼接成一张大图。
    order: 'asc' 表示按照原顺序（升序），'desc' 表示倒序排列。
    """
    if order == 'desc':
        tensor_np = tensor_np[::-1]  # 倒序
    # list 中每个元素形状为 (H, W)
    return np.concatenate([img for img in tensor_np], axis=0)

# 假设 input_tensor, gt_tensor, output_tensor 均为 torch.Tensor，形状为 (B, T, H, W)，B=1
# 若它们已经是 numpy 数组，则可直接使用。
input_tensor = input_frames
gt_tensor = target_frames
# output_tensor = gen_result[:,:,0]
output_tensor = gen_result

T_in = args.input_length
T_out = args.evo_ic

# 假设 input_tensor, gt_tensor, output_tensor 均存在，形状为 (B, T, H, W)，B=1
# 这里仅作示例，实际使用时请加载你的 tensor 数据
# 例如：input_tensor = torch.rand(1, T_in, H, W)
# 这里我们直接假设它们已经存在，并转换为 numpy 数组：
input_np = input_tensor[0].detach().cpu().numpy()   # shape: (T_in, H, W)
gt_np = gt_tensor[0].detach().cpu().numpy()           # shape: (T_out, H, W)
output_np = output_tensor[0].detach().cpu().numpy()   # shape: (T_out, H, W)

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

# 创建图形和子图
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
# 确保 axes 是二维数组
if nrows == 1:
    axes = np.expand_dims(axes, axis=0)
if ncols == 1:
    axes = np.expand_dims(axes, axis=1)

loss_pool_list = []
for i in range(T_out):
    loss_pool = pool_regularization(gt_tensor.unsqueeze(2)[:,i:i+1], [output_tensor.unsqueeze(2)[:,i:i+1]])
    loss_pool_list.append(loss_pool.item())
loss_pool_worst = []
for i in range(T_out):
    loss_pool = pool_regularization(gt_tensor.unsqueeze(2)[:,i:i+1], [input_tensor.unsqueeze(2)[:,-1:]])
    loss_pool_worst.append(loss_pool.item())

# 遍历每一行和每一列绘制图像
for r in range(nrows):
    # 第一列：Input，时间步按降序排列（从上到下 t 从大到小）
    if r < T_in:
        t_idx = T_in - 1 - r  # 降序：上面显示最大的 t
        masked_data = get_masked_array(input_np[t_idx])
        im = axes[r, 0].imshow(masked_data, cmap=cmap_input)
        axes[r, 0].set_title(f"Input t={t_idx}")
        fig.colorbar(im, ax=axes[r, 0])
    else:
        axes[r, 0].axis('off')

    # 第二列：GT，时间步按升序排列（从上到下 t 从小到大）
    if r < T_out:
        t_idx = r
        masked_data = get_masked_array(gt_np[t_idx])
        im = axes[r, 1].imshow(masked_data, cmap=cmap_gt)
        axes[r, 1].set_title(f"GT t={t_idx}")
        fig.colorbar(im, ax=axes[r, 1])
    else:
        axes[r, 1].axis('off')

    # 第三列：Output，时间步按升序排列（从上到下 t 从小到大）
    if r < T_out:
        t_idx = r
        masked_data = get_masked_array(output_np[t_idx])
        im = axes[r, 2].imshow(masked_data, cmap=cmap_output)
        axes[r, 2].set_title(f"Output t={t_idx}")
        fig.colorbar(im, ax=axes[r, 2])
    else:
        axes[r, 2].axis('off')

    # 第四列：Diff（GT与Output的差值），时间步按升序排列（从上到下 t 从小到大）
    if r < T_out:
        t_idx = r
        masked_data = get_masked_array(diff_np[t_idx])
        im = axes[r, 3].imshow(masked_data, cmap=cmap_diff)
        axes[r, 3].set_title(f"Diff loss={loss_pool_list[t_idx]:.2f}, k={loss_pool_list[t_idx]/loss_pool_worst[t_idx]:.2f}")
        fig.colorbar(im, ax=axes[r, 3])
    else:
        axes[r, 3].axis('off')

# 调整布局并显示图形
plt.tight_layout()
plt.show()