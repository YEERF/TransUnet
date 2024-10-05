import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()  # 调用父类构造函数

        self.head_num = head_num  # 多头的数量
        self.dk = (embedding_dim // head_num) ** (1 / 2)  # 缩放因子，用于缩放点积注意力

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)  # 线性层，用于生成查询（Q）、键（K）和值（V）
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)  # 输出线性层

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)  # 通过线性层生成Q、K、V

        query, key, value = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num))  # 将Q、K、V重塑为多头注意力的格式
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk  # 计算点积注意力的能量

        if mask is not None:  # 如果提供了掩码，则在能量上应用掩码
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)  # 应用softmax函数，得到注意力权重

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)  # 应用注意力权重到值上

        x = rearrange(x, "b h t d -> b t (h d)")  # 重塑x以准备输出
        x = self.out_attention(x)  # 通过输出线性层

        return x


# 定义MLP模块
class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()  # 调用父类构造函数

        self.mlp_layers = nn.Sequential(  # 定义MLP的层
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),  # GELU激活函数
            nn.Dropout(0.1),  # Dropout层，用于正则化
            nn.Linear(mlp_dim, embedding_dim),  # 线性层
            nn.Dropout(0.1)  # Dropout层
        )

    def forward(self, x):
        x = self.mlp_layers(x)  # 通过MLP层
        return x


# 定义Transformer编码器块
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()  # 调用父类构造函数

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)  # 多头注意力模块
        self.mlp = MLP(embedding_dim, mlp_dim)  # MLP模块

        self.layer_norm1 = nn.LayerNorm(embedding_dim)  # 第一层归一化
        self.layer_norm2 = nn.LayerNorm(embedding_dim)  # 第二层归一化

        self.dropout = nn.Dropout(0.1)  # Dropout层

    def forward(self, x):
        _x = self.multi_head_attention(x)  # 通过多头注意力模块
        _x = self.dropout(_x)  # 应用dropout
        x = x + _x  # 残差连接
        x = self.layer_norm1(x)  # 第一层归一化

        _x = self.mlp(x)  # 通过MLP模块
        x = x + _x  # 残差连接
        x = self.layer_norm2(x)  # 第二层归一化

        return x


# 定义Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()  # 调用父类构造函数

        self.layer_blocks = nn.ModuleList([  # 创建一个模块列表，包含多个编码器块
            TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)
        ])

    def forward(self, x):
        for layer_block in self.layer_blocks:  # 遍历每个编码器块
            x = layer_block(x)  # 通过每个块
        return x


# 定义ViT模型
class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim,
                 classification=True, num_classes=1):
        super().__init__()  # 调用父类构造函数

        self.patch_dim = patch_dim  # 定义patch的维度
        self.classification = classification  # 是否进行分类
        self.num_tokens = (img_dim // patch_dim) ** 2  # 计算tokens的数量
        self.token_dim = in_channels * (patch_dim ** 2)  # 计算每个token的维度

        self.projection = nn.Linear(self.token_dim, embedding_dim)  # 线性层，用于将patches投影到embedding空间
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))  # 可学习的embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  # 类别token

        self.dropout = nn.Dropout(0.1)  # Dropout层

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)  # Transformer编码器

        if self.classification:  # 如果是分类任务
            self.mlp_head = nn.Linear(embedding_dim, num_classes)  # 分类头

    def forward(self, x):
        img_patches = rearrange(x,  # 将输入图像重塑为patches序列
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape  # 获取批次大小、tokens数量和通道数

        project = self.projection(img_patches)  # 将patches投影到embedding空间
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)  # 重复cls_token以匹配批次大小

        patches = torch.cat((token, project), dim=1)  # 将cls_token和投影后的patches拼接
        patches += self.embedding[:tokens + 1, :]  # 将可学习的embedding添加到patches

        x = self.dropout(patches)  # 应用dropout
        x = self.transformer(x)  # 通过Transformer编码器
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]  # 如果是分类任务，使用cls_token的输出；否则，使用patches的输出

        return x