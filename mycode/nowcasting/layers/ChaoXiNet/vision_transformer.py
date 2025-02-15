# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from mycode.nowcasting.layers.SwinUnet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from mycode.nowcasting.layers.SwinUnet.config import _C as config

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=10, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")




# import yaml
# with open('configs/swin_tiny_patch4_window7_224_lite.yaml', 'r', encoding='utf-8') as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)

# print(config)
net = swin_unet = SwinTransformerSys(img_size=256,
                                patch_size=4,
                                in_chans=10,
                                num_classes=10,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=8,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
# input = torch.randn(3,10,256,256)
# out = net(input)
# print(out.shape)

import torchview
# net = SwinTransformerV2().to('cuda')
# out = net(torch.randn(1, 10, 256, 256).to('cuda'))
# print(out.shape)

dummy_input = torch.randn(16, 10, 256, 256).cuda()



print("\n>>> [1] 使用 torchview 可视化网络结构图 ...")
graph = torchview.draw_graph(
    net,
    input_size=dummy_input.shape,
    expand_nested=True,
    save_graph=False,  # 先不让其自动保存
    roll=False,
    hide_inner_tensors=True,
    graph_dir='UD'
)

dot = graph.visual_graph
# 设置 DPI
dot.graph_attr['dpi'] = '100'
# dot.attr(size="3,100!")
dot.attr(ratio="auto")

dot.render(
    filename="swin_unet",  # 指定输出文件名（无后缀）
    directory="./",        # 输出目录
    cleanup=True,          # render后删除多余的临时文件
    format="png"           # 导出 png 格式
)

graph.visual_graph
# 渲染并保存为 PDF（也可以保存为 SVG 等格式）
graph.visual_graph.attr(size="10000,20000!")
# graph.visual_graph.attr(ranksep="0.5")
# graph.render(format='pdf', cleanup=True)
print("已保存 torchview 可视化结构图到 model.pdf")


dot = graph.visual_graph
# dot.attr(rankdir="LR")  # 改为水平布局
dot.attr(size="10000,40000!")  # 调整 size
dot.attr(ratio="compress")
dot.render(
    filename="swin_unet",  # 输出文件名（不含后缀）
    format="png",  # 输出格式
    directory=".",  # 输出到当前文件夹
    cleanup=True  # 渲染后删除中间文件
)
print("已保存 torchview 可视化结构图到 model_torchview.pdf")

