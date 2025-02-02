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

# ===== 如果你需要下面的可视化功能，请确保已安装 =====
# pip install torchview netron
import torchview
import torchvision
import netron

# =========== 模型相关的依赖 =============
# 假设你已有 nowcasting.data_provider.datasets_factory & nowcasting.models.nowcastnet
# (若文件结构不同，可自行修改引用)
from mycode.nowcasting.data_provider import datasets_factory
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
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--cpu_worker', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--dataset_path', type=str, default='data/dataset/mrms/figure')
parser.add_argument('--model_name', type=str, default='NowcastNet')
parser.add_argument('--pretrained_model', type=str, default='data/checkpoints/mrms_model.ckpt')
parser.add_argument('--gen_frm_dir', type=str, default='results/us/')
parser.add_argument('--case_type', type=str, default='normal')
parser.add_argument('--input_length', type=int, default=9)
parser.add_argument('--total_length', type=int, default=29)
parser.add_argument('--img_height', type=int, default=512)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_ch', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_save_samples', type=int, default=10)
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

args.pool_loss_k = 4
# 创建输出文件夹（若已存在则删除重建）
if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print('>>> 初始化并加载模型 ...')

# network = Net(args).to(args.device)
# stats = torch.load('../data/checkpoints/mrms_model.ckpt')
# network.load_state_dict(stats)

EvolutionNet = Evolution_Network(args.input_length, args.pred_length, base_c=32).to(args.device)
GenerativeEncoder = Generative_Encoder(args.total_length, base_c=args.ngf).to(args.device)
GenerativeDecoder = Generative_Decoder(args).to(args.device)
NoiseProjector = Noise_Projector(args.ngf, args).to(args.device)
TemporalDiscriminator = Temporal_Discriminator(args).to(args.device)


# model = Model(args)

batch_size_test = 16
test_input_handle = datasets_factory.data_provider(args)
args.batch_size = batch_size_test

for batch_id, test_ims in enumerate(test_input_handle):
    test_ims = test_ims['radar_frames'].numpy()
    test_ims = torch.FloatTensor(test_ims).to(args.device)
    print('input size:    ', test_ims.shape)

    # img_gen = network(test_ims)


    test_ims = test_ims[:, :, :, :, :1]
    # B, T, H, W, C

    frames = test_ims.permute(0, 1, 4, 2, 3)
    # B, T, C, H, W
    batch = frames.shape[0]
    height = frames.shape[3]
    width = frames.shape[4]

    # Input Frames
    input_frames = frames[:, :args.input_length]
    target_frames = frames[:, args.input_length:,0]

    input_frames = input_frames.reshape(batch, args.input_length, height, width)

    # Evolution Network
    intensity, motion = EvolutionNet(input_frames)
    motion_ = motion.reshape(batch, args.gen_oc, 2, height, width)
    intensity_ = intensity.reshape(batch, args.pred_length, 1, height, width)
    series = []
    series_bili = []
    last_frames = test_ims[:, (args.input_length - 1):args.input_length, :, :, 0]
    sample_tensor = torch.zeros(1, 1, args.img_height, args.img_width)
    grid = make_grid(sample_tensor)
    grid = grid.repeat(batch, 1, 1, 1)
    for i in range(args.pred_length):
        last_frames_bili = warp(last_frames, motion_[:, i], grid.cuda(), mode="bilinear", padding_mode="border")
        last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
        last_frames = last_frames + intensity_[:, i]
        series.append(last_frames)
        series_bili.append(last_frames_bili)
    evo_result = torch.cat(series, dim=1)
    evo_result = evo_result / 128

    evo_result_bili = torch.cat(series_bili, dim=1)
    evo_result_bili = evo_result_bili / 128
    # todo 阻止梯度累计
    loss_evo = evolution_loss(
        pred_final=evo_result,
        pred_bili=evo_result_bili,  # 如果有bilinear结果也可加
        real=frames[:, args.input_length:,0],  # [B,T,H,W]
        motion=motion_,  # [B,T,2,H,W]
        lam=0.01
    )


    # Generative Network
    evo_feature = GenerativeEncoder(torch.cat([input_frames, evo_result], dim=1))

    gen_result_list = []
    dis_result_pre_list = []
    for _ in range(args.pool_loss_k):
        noise = torch.randn(batch, args.ngf, height // 32, width // 32).cuda()
        noise_feature = NoiseProjector(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1,
                                                                                                          height // 8,
                                                                                                          width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = GenerativeDecoder(feature, evo_result)
        gen_result = gen_result.unsqueeze(2)
        gen_result_list.append(gen_result)

        # print(gen_result.shape)

        # Temporal Discriminator
        dis_result_pre = TemporalDiscriminator(gen_result, input_frames.unsqueeze(2))
        dis_result_pre_list.append(dis_result_pre)
    dis_result_GT = TemporalDiscriminator(target_frames.unsqueeze(2), input_frames.unsqueeze(2))

    loss_adv = adversarial_loss(dis_result_pre_list)
    loss_pool = pool_regularization(target_frames.unsqueeze(2), gen_result_list)
    loss_generative = 6*loss_adv + 20*loss_pool

    loss_disc = discriminator_loss(dis_result_GT, dis_result_pre_list)

    break


# print('>>> 开始测试推理 ...')
# test_wrapper_pytorch_loader(model, args)
# print('>>> 推理完成! 结果已保存在 "{}" 下'.format(args.gen_frm_dir))
#
# # 如果指定可视化，就绘制网络结构
# if args.visualize:
#     print('\n>>> 准备进行模型可视化 ...')
#     visualize_model(model, args)



