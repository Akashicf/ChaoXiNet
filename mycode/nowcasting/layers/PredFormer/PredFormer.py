import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)


class MultiHeadAttention(nn.Module):
    """与原始代码类似的多头注意力"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        假设输入 x 形状为 [B, seq_len, dim]
        """
        b, seq_len, dim = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # [B, seq_len, heads*dim_head] -> (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, heads, n, n]
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [b, heads, n, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入形状 [B, seq_len, dim]
        SwiGLU: out = (xW1) * Swish(xW2)
        这里只是用一种简单实现示例
        """
        x = self.norm(x)

        # 先做两次线性投影
        x1 = self.fc1(x)  # 用于做“门”
        x2 = self.fc2(x)  # 用于做“激活”

        # Swish 激活: swish(x) = x * sigmoid(x)，这里可用 F.silu(x)
        gated = x1 * nn.SiLU()(x2)  # 或者 x1 * F.silu(x2), 参考官方 SwiGLU

        # Dropout -> Linear -> Dropout
        gated = self.dropout1(gated)
        out = self.linear_out(gated)
        out = self.dropout2(out)

        return out


class QuadrupletSTTSBlock(nn.Module):
    """
    在同一层内依次执行四次注意力:
    1) S-Attn
    2) T-Attn
    3) T-Attn
    4) S-Attn
    每次注意力后接 SwiGLU FFN
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        # 空间注意力
        self.s_attn1 = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.s_ffn1 = SwiGLUFeedForward(dim, mlp_dim, dropout)

        self.t_attn1 = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.t_ffn1 = SwiGLUFeedForward(dim, mlp_dim, dropout)

        self.t_attn2 = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.t_ffn2 = SwiGLUFeedForward(dim, mlp_dim, dropout)

        self.s_attn2 = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.s_ffn2 = SwiGLUFeedForward(dim, mlp_dim, dropout)

    def forward(self, x, T, N):
        """
        x 形状: [B, T*N, D]
        参数 T: 时间帧数
        参数 N: 空间 patch 数 (H/p * W/p)
        """
        B, TN, D = x.shape
        assert TN == T * N, "输入序列长度应当为 T*N"

        # --- Step 1: S-Attn ---
        # reshape 到 [B, T, N, D]，再把 (T, N) 合并成 seq_len = T*N 来送入注意力
        x_s1 = rearrange(x, 'b (t n) d -> b t n d', t=T, n=N)
        # 在这里选择“空间维”作为 seq_len 处理: 先把 t 并进 batch 或者 transpose(t, n) 都可以
        x_s1 = rearrange(x_s1, 'b t n d -> b (t n) d')

        s1_out = self.s_attn1(x_s1) + x_s1
        s1_out = self.s_ffn1(s1_out) + s1_out
        # reshape 回 [B, T*N, D]
        s1_out = rearrange(s1_out, 'b (t n) d -> b (t n) d', t=T, n=N)

        # --- Step 2: T-Attn ---
        # 这里要在时间维做注意力，reshape 成 [B, T, N, D]，把 N 合并到 batch 或者做 transpose
        x_t1 = rearrange(s1_out, 'b (t n) d -> b t n d', t=T, n=N)
        x_t1 = rearrange(x_t1, 'b t n d -> b (n t) d')  # 合并 t 到 seq_len，也可 transpose 来实现
        t1_out = self.t_attn1(x_t1) + x_t1
        t1_out = self.t_ffn1(t1_out) + t1_out
        # reshape 回 [B, T*N, D]
        t1_out = rearrange(t1_out, 'b (n t) d -> b (t n) d', t=T, n=N)

        # --- Step 3: 再来一次 T-Attn ---
        x_t2 = rearrange(t1_out, 'b (t n) d -> b t n d', t=T, n=N)
        x_t2 = rearrange(x_t2, 'b t n d -> b (n t) d')
        t2_out = self.t_attn2(x_t2) + x_t2
        t2_out = self.t_ffn2(t2_out) + t2_out
        t2_out = rearrange(t2_out, 'b (n t) d -> b (t n) d', t=T, n=N)

        # --- Step 4: 最后一次 S-Attn ---
        x_s2 = rearrange(t2_out, 'b (t n) d -> b t n d', t=T, n=N)
        x_s2 = rearrange(x_s2, 'b t n d -> b (t n) d')
        s2_out = self.s_attn2(x_s2) + x_s2
        s2_out = self.s_ffn2(s2_out) + s2_out

        return s2_out  # [B, T*N, D]


def build_3d_sin_pos_encoding(T, H_blocks, W_blocks, dim):
    """
    为 (t, h, w) 生成一个 dim 维的正弦位置编码 (absolute positional encoding).
    最终返回形状: [1, T * (H_blocks * W_blocks), dim].
    """

    # 创建存放结果的 tensor
    pe = torch.zeros(T, H_blocks, W_blocks, dim)

    # 为了简化，这里假设对 (t, h, w) 这三维分别使用不同频率的正余弦编码
    # freq_base 用于控制频率跨度，可根据需要调整
    freq_base = 10000.0

    # t -> 第 0 维
    for t_idx in range(T):
        for h_idx in range(H_blocks):
            for w_idx in range(W_blocks):
                # 计算位置 (t_idx, h_idx, w_idx)
                # 构造 pos = [t_idx, h_idx, w_idx] 再映射到 [0, 1, 2, ...] 维度
                # 这里只是示例，可以细化为对 dim 各自编码
                pos = torch.tensor([t_idx, h_idx, w_idx], dtype=torch.float32)

                # 对每个通道 c (0 ~ dim-1)，用正余弦函数
                for c in range(dim):
                    div_term = (c // 6)  # 这里只是演示，每 6 维换一种频率等
                    # 也可以根据 c 在整段维度上的位置算频率
                    phase = pos[c % 3]  # 取 t/h/w
                    # 常见做法： sin(pos / (freq_base^(2i/dim))) + cos(...)
                    # 这里仅做最简单的示例
                    if c % 2 == 0:
                        pe[t_idx, h_idx, w_idx, c] = math.sin(phase / (freq_base ** div_term))
                    else:
                        pe[t_idx, h_idx, w_idx, c] = math.cos(phase / (freq_base ** div_term))

    # reshape 成 [T * N, dim], 并添加 batch 维度 1
    pe = rearrange(pe, 't h w d -> 1 (t h w) d')
    return pe


class QuadrupletSTTSNet(nn.Module):
    """
    演示性的网络结构:
    1) patch embedding
    2) 堆叠 N 个 Quadruplet-STTS block
    3) 将最终特征投影回 [B, C, H, W]
    """

    def __init__(
            self,
            image_size=64,  # 图像高宽
            patch_size=8,
            in_channels=1,  # 输入图像通道数
            dim=128,  # patch embedding 后的维度
            depth=2,  # 堆叠多少个 Quadruplet-STTS block
            heads=4,
            dim_head=32,
            mlp_dim=256,
            dropout=0.0,
            T=10  # 时间帧长度
    ):
        super().__init__()
        # 计算 H_blocks, W_blocks
        self.image_height = image_size
        self.image_width = image_size
        self.patch_height = patch_size
        self.patch_width = patch_size

        self.H_blocks = self.image_height // self.patch_height
        self.W_blocks = self.image_width // self.patch_width

        self.T = T
        self.num_patches = self.H_blocks * self.W_blocks
        self.patch_dim = in_channels * self.patch_height * self.patch_width
        self.dim = dim

        # 1) patch embedding
        self.patch_embed = nn.Sequential(
            # 将 (B, C, T, H, W) -> (B, T, N, patch_dim)
            # 其中 N = H_blocks * W_blocks
            Rearrange('b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)',
                      p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim)
        )

        # 2) Quadruplet-STTS blocks (此处仅演示，假设已在其他文件中定义好)
        Modules = []
        for _ in range(depth):
            Modules.append(QuadrupletSTTSBlock(dim, heads, dim_head, mlp_dim, dropout=dropout))
        self.blocks = nn.ModuleList(Modules)

        # 3) 输出层：将维度投影回原图
        self.to_img = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.patch_dim)
        )

        self.final_reshape = Rearrange(
            'b t (h w) (p1 p2 c) -> b c t (h p1) (w p2)',
            p1=self.patch_height,
            p2=self.patch_width,
            c=in_channels,
            # 需要显式告诉 einops, 这里的 n = h*w
            h=self.H_blocks,
            w=self.W_blocks
        )

        # ============= 新增：位置编码 =============
        # 这里以 "绝对正弦位置编码" 为例
        self.register_buffer(
            'pos_embed',
            build_3d_sin_pos_encoding(self.T, self.H_blocks, self.W_blocks, dim),
            persistent=False
        )
        # =========================================

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        输出: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        # 1) patch embedding -> [B, T, N, dim], 再 reshape 到 [B, T*N, dim]
        x = self.patch_embed(x)  # [B, T, N, dim]
        x = rearrange(x, 'b t n d -> b (t n) d')

        # 2) 加上位置编码 [1, T*N, dim]
        #   注意如果输入 batch 较大，可 repeat 到 [B, T*N, dim]
        x = x + self.pos_embed[..., :].expand(B, -1, -1)

        # 3) 逐层 STTS
        for block in self.blocks:
            x = block(x, self.T, self.H_blocks*self.W_blocks)  # 这里假设 block forward 不需要额外参数

        # 4) 恢复图像尺寸
        x = self.to_img(x)  # [B, T*N, patch_dim]
        x = rearrange(
            x,
            'b (t n) pd -> b t n pd',
            t=T,
            n=self.num_patches
        )
        out = self.final_reshape(x)  # [B, C, T, H, W]
        return out

B, C, T, H, W = 2, 1, 10, 64, 64
x = torch.randn(B, C, T, H, W)

net = QuadrupletSTTSNet(
    image_size=64,
    patch_size=8,
    in_channels=1,
    dim=128,
    depth=2,       # 堆叠 2 个 Quadruplet-STTS block
    heads=4,
    dim_head=32,
    mlp_dim=256,
    dropout=0.1,
    T=10
)

out = net(x)
print("输入:", x.shape, "输出:", out.shape)


import torchview

dummy_input = torch.randn(2, 1, 10, 64, 64)



print("\n>>> [1] 使用 torchview 可视化网络结构图 ...")
graph = torchview.draw_graph(net, input_size=dummy_input.shape, expand_nested=True, save_graph=True,
                             filename="PerdFormer", roll=False, hide_inner_tensors=True, graph_dir='UD')

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
    filename="model_torchview",  # 输出文件名（不含后缀）
    format="png",  # 输出格式
    directory=".",  # 输出到当前文件夹
    cleanup=True  # 渲染后删除中间文件
)
print("已保存 torchview 可视化结构图到 model_torchview.pdf")