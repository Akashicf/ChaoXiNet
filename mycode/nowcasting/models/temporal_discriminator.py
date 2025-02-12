import torch
import torch.nn as nn
from mycode.nowcasting.layers.utils import spectral_norm


class Temporal_Discriminator(nn.Module):
    def __init__(self, opt):
        super(Temporal_Discriminator, self).__init__()
        """
        假设:
          opt.input_length -> T_in
          opt.evo_ic       -> T_out
        """
        self.T_in = opt.input_length*opt.input_channel
        self.T_out = opt.evo_ic

        # -------------------
        # 第一条路径: 2D 卷积
        # in_channels = T_in, out_channels = out_channels_1
        # -------------------
        self.out_channels_1 = 64
        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channels=self.T_in + self.T_out,
                out_channels=self.out_channels_1,
                kernel_size=9,
                stride=2,
                padding=9 // 2
            )
        )

        # -------------------
        # 第二条路径: 3D 卷积
        # in_channels = 1, out_channels=4, kernel=(T_in,9,9), stride=(1,2,2)
        # 最终会展平成 out_channels_2 = 4*(some_time_size)
        # 如果 time 方向被完全卷积到 1, 则 out_channels_2=4
        # -------------------
        self.conv2 = spectral_norm(
            nn.Conv3d(
                in_channels=1,
                out_channels=4,
                kernel_size=(self.T_in, 9, 9),
                stride=(1, 2, 2),
                padding=(0, 9 // 2, 9 // 2)
            )
        )
        # 根据需要, 您可以根据 padding/stride 计算最后的实际通道
        # 这里示例把它称为 out_channels_2:
        self.out_channels_2 = 4 * (self.T_out+1)  # 如果 time 卷到 1, 即 4*(1)

        # -------------------
        # 第三条路径: 3D 卷积
        # in_channels=1, out_channels=8, kernel=(T_out,9,9), stride=(1,2,2)
        # 最终展平 => out_channels_3 = 8*(some_time_size)
        # -------------------
        self.conv3 = spectral_norm(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(self.T_out, 9, 9),
                stride=(1, 2, 2),
                padding=(0, 9 // 2, 9 // 2)
            )
        )
        self.out_channels_3 = 8* (self.T_in+1)  # 同理

        # 最终拼接通道数
        concat_channels = self.out_channels_1 + self.out_channels_2 + self.out_channels_3

        # ----------------------------------------------------------------
        # 下面构造 4 个 LBlock
        # 第1次: in_channel=concat_channels -> out_channel=128, stride=2
        # 第2次: 128 -> 256, stride=2
        # 第3次: 256 -> 512, stride=2
        # 第4次: 512 -> 512, stride=1 (不再减 spatial 尺寸)
        # ----------------------------------------------------------------
        self.lblock1 = DoubleConv(concat_channels, 128, stride=2)
        self.lblock2 = DoubleConv(128, 256, stride=2)
        self.lblock3 = DoubleConv(256, 512, stride=2)
        self.lblock4 = DoubleConv(512, 512, stride=1)

        self.BN = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=1e-2)
        self.conv_out = spectral_norm(nn.Conv2d(512, 1, kernel_size=3, padding=1, stride=1))


    def forward(self, x_bar, x_input):
        """
        x_input.shape = [B, T, C, H, W]
        x_bar.shape   = [B, T2, C, H, W]
        其中:
          - 在时间维 x_input 拿前 self.T_in 帧
          - x_bar 拿前 self.T_out 帧
          - C=1 或多通道时, 这里只示例取第 0 通道
        """
        x = torch.cat([x_bar, x_input], dim=1)
        B, T, C, H, W = x_input.shape
        # 1) 取前 T_in 帧并做 2D 卷积
        #    conv1 期望输入是 [B, T_in, H, W]
        x_in2d = x[:, :,0]  # -> [B, T_in, H, W]
        feat1 = self.conv1(x_in2d)              # -> [B, out_channels_1, H/2, W/2]

        # 2) 再对同样的 x_in2d 做 3D 卷积
        #    conv2 期望 [B, 1, T_in, H, W]
        x_in3d = x_in2d.unsqueeze(1)            # -> [B, 1, T_in, H, W]
        feat2 = self.conv2(x_in3d)             # -> [B, 4, 1, H/2, W/2] (若time维度卷积到1)
        feat2 = feat2.reshape(B, self.out_channels_2, H//2, W//2)              # -> [B, 4, H/2, W/2]

        # 3) 对 x_bar 做 3D 卷积
        feat3 = self.conv3(x_in3d)            # -> [B, 8, 1, H/2, W/2]
        feat3 = feat3.reshape(B, self.out_channels_3, H//2, W//2)              # -> [B, 8, H/2, W/2]

        # 4) 拼接通道 => [B, out_channels_1 + out_channels_2 + out_channels_3, H/2, W/2]
        feats = torch.cat([feat1, feat2, feat3], dim=1)

        # 5) 依次通过 4 个 LBlock
        #    尺寸变化:
        #      - lblock1: -> [B, 128, H/4, W/4]
        #      - lblock2: -> [B, 256, H/8, W/8]
        #      - lblock3: -> [B, 512, H/16, W/16]
        #      - lblock4: -> [B, 512, H/16, W/16]
        out = self.lblock1(feats)
        out = self.lblock2(out)
        out = self.lblock3(out)
        out = self.lblock4(out)

        # 这里您可以根据需求返回判别器的“打分”或者特征
        # 若需要一个标量或向量, 还可以在最后做一次全局池化/全连接等
        out = self.BN(out)
        out = self.leaky_relu(out)
        out = self.conv_out(out)
        return out


class DoubleConv(nn.Module):
    def   __init__(self, in_channels, out_channels, kernel=3, stride=2, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=kernel//2, stride=stride)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=kernel//2, stride=1)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel//2, stride=stride))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)
        x = x + shortcut
        return x
