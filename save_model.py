#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

from mycode.nowcasting.data_provider.npz_loader import *
from mycode.nowcasting.layers.SwinUnet.vision_transformer import SwinUnet

from mycode.nowcasting.layers.PredFormer.PredFormer import *

# 打开cudnn 加速
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if 'pydevconsole.py' in sys.argv[0]:
    sys.argv = sys.argv[:1]

parser = argparse.ArgumentParser(description='NowcastNet Inference & Visualization')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--worker', type=int, default=16)
parser.add_argument('--cpu_worker', type=int, default=16)
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--dataset_path', type=str, default='../data/dataset/mrms/figure')
parser.add_argument('--dataset_path_train', type=str, default='../data/dataset/yangben_solo_h5')
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
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_save_samples', type=int, default=1000)
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--visualize', action='store_true',
                    help='是否使用 torchview 和 netron 进行模型结构可视化')
parser.add_argument('--ts', action='store_true', default=False)

args, unknown = parser.parse_known_args()


# 输出长度
args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.pred_length = args.total_length - args.input_length

args.ic_feature = args.ngf * 10

args.pool_loss_k = 2
args.use_num = 10
args.checkpoint_path = '../result/PredFormer'
data_dir = '../data/dataset/yangben_all'
data_dir_h5 = '../data/dataset/yangben_solo_h5'

# merged_data_dir = '../data/dataset/merged_data.npz'
args.experiment = 'run3'
# hyperparameters
# evo_net
alpha = 0.01
lr_evo = 200e-5
lr_decrease_epoch = 40
lr_stop_epoch = 500
lr_evo_de = 20e-5

# generator
lr_gen = 3e-5
lr_gen_de = 1e-5
beta = 6
gamma = 0.5

# TD
lr_disc = 3e-5

reg_loss = True
value_lim=[0,65]

num_epochs = 500
# batch_size = 1
use_amp = True


print('>>> 初始化并加载模型 ...')

class SwinUnetMotion(nn.Module):
    def __init__(self, config):
        super(SwinUnetMotion, self).__init__()
        self.motion_net = SwinUnet(config, num_classes=20)
        self.intensity_net = SwinUnet(config, num_classes=10)
    def forward(self, input):
        return self.intensity_net(input), self.motion_net(input)


EvolutionNet = QuadrupletSTTSNet(
    image_size=256,
    patch_size=16,
    in_channels=1,
    dim=512,
    depth=6,
    heads=8,
    dim_head=24,
    mlp_dim=1024,
    dropout=0.1,
    dropout_path=0.25,
    T=10
)

checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment)
os.makedirs(checkpoint_dir, exist_ok=True)

# 尝试搜集已有 ckpt
ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
ckpts.sort()  # 根据名称排序，最后一个视作最新


if len(ckpts) > 0:
    latest_ckpt = ckpts[-1]
    print(f"[INFO] Found checkpoint: {latest_ckpt}, now loading...")
    checkpoint = torch.load(latest_ckpt, map_location=args.device)

    latest_ckpt = ckpts[-1]
    # 加载模型参数
    EvolutionNet.load_state_dict(checkpoint['EvolutionNet'])

    # 恢复 epoch 及迭代数
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}, global_step={global_step}")

else:
    print("[INFO] No existing checkpoint found, starting from scratch...")
    latest_ckpt = None


ckpt_path = os.path.join(checkpoint_dir, f"ckpt_test_step{global_step+1:06d}.pth")
torch.save({
    'epoch': start_epoch,
    'global_step': global_step,
    'train_losses': train_losses,
    'EvolutionNet': EvolutionNet.state_dict(),
    'test_losses': test_losses,
}, ckpt_path)
print(f"[INFO] Saved checkpoint to {ckpt_path}")