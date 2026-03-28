"""
LS卷积模块 (LS Convolution)
来自LSNet: See Large + Focus Small
大核深度卷积捕获全局结构 + 小核动态卷积聚焦局部细节
特别适合表情识别，增强五官特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DepthWiseConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, in_channels, 1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DynamicConv2d(nn.Module):
    """
    分组动态卷积（轻量化版）
    替代4个独立卷积核，使用分组卷积大幅降低参数量
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 num_groups: int = 16):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            num_groups: 分组数（G=16时参数量约为原来的1/4）
        """
        super().__init__()
        self.num_groups = num_groups

        # 使用分组卷积替代多个独立卷积
        self.dynamic_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2,
            groups=num_groups,  # 关键：使用分组卷积
            bias=False
        )

        # 轻量级注意力：直接生成分组权重（LKP）
        # 不使用独立的注意力网络
        self.lkp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_groups, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LKP直接生成分组权重
        group_weights = self.lkp(x)  # (B, num_groups, 1, 1)

        # 分组卷积
        x = self.dynamic_conv(x)

        # 应用分组权重
        # 将通道按group维度切分并应用权重
        B, C, H, W = x.shape
        x = x.view(B, self.num_groups, -1, H, W)  # (B, G, C//G, H, W)
        group_weights = group_weights.unsqueeze(2)  # (B, G, 1, 1, 1)
        x = x * group_weights
        x = x.view(B, C, H, W)  # 还原形状

        return x


class LSConv(nn.Module):
    """
    LS卷积模块：See Large + Focus Small（轻量化版）

    结构:
    1. 大核深度卷积（See Large）：捕获全局上下文（7×7）
    2. 小核动态卷积（Focus Small）：聚焦局部细节（3×3，G=16）
    3. 特征融合
    """

    def __init__(self,
                 channels: int,
                 large_kernel: int = 7,  # 从21改为7
                 small_kernel: int = 3,
                 num_groups: int = 16):  # 使用分组卷积
        """
        Args:
            channels: 通道数
            large_kernel: 大核尺寸（从21改为7）
            small_kernel: 小核尺寸（3）
            num_groups: 动态卷积的分组数（G=16）
        """
        super().__init__()

        # See Large: 大核深度卷积捕获全局结构
        # 从21×21改为7×7，参数量减少约9倍
        self.see_large = DepthWiseConv(
            channels, large_kernel,
            padding=large_kernel // 2
        )

        # Focus Small: 小核分组动态卷积聚焦局部细节
        # 使用分组卷积（G=16），参数量约为原来的1/4
        self.focus_small = DynamicConv2d(
            channels, channels,
            small_kernel,
            num_groups=num_groups
        )

        # 特征融合（轻量化）
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),  # 1×1卷积
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            输出特征 (B, C, H, W)
        """
        identity = x

        # See Large: 全局感知
        large_feat = self.see_large(x)

        # Focus Small: 局部聚焦
        small_feat = self.focus_small(x)

        # 特征融合
        fused = torch.cat([large_feat, small_feat], dim=1)
        fused = self.fusion(fused)

        # 残差连接
        output = fused + identity

        return output


class LSConvBlock(nn.Module):
    """
    LS卷积块：包含多个LSConv的堆叠
    用于在Transformer Encoder之前进行局部特征增强
    """
    
    def __init__(self, 
                 channels: int,
                 num_layers: int = 2,
                 large_kernel: int = 21,
                 small_kernel: int = 3):
        """
        Args:
            channels: 通道数
            num_layers: LSConv层数
            large_kernel: 大核尺寸
            small_kernel: 小核尺寸
        """
        super().__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.append(LSConv(channels, large_kernel, small_kernel))
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


# 测试代码
if __name__ == "__main__":
    print("测试LS卷积模块...")
    
    # 测试LSConv
    x = torch.randn(2, 64, 56, 56)
    ls_conv = LSConv(channels=64)
    out = ls_conv(x)
    print(f"LSConv输入: {x.shape}, 输出: {out.shape}")
    
    # 测试LSConvBlock
    ls_block = LSConvBlock(channels=64, num_layers=2)
    out_block = ls_block(x)
    print(f"LSConvBlock输入: {x.shape}, 输出: {out_block.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in ls_conv.parameters())
    print(f"LSConv参数量: {total_params:,}")
    
    # 测试前向传播
    print("\n前向传播测试:")
    with torch.no_grad():
        out = ls_conv(x)
        print(f"输出均值: {out.mean().item():.4f}, 标准差: {out.std().item():.4f}")
    
    print("\nLS卷积模块测试通过!")
