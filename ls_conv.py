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
    动态卷积：根据输入特征动态调整卷积核
    实现Focus Small思想
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 num_kernels: int = 4):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            num_kernels: 动态卷积核数量
        """
        super().__init__()
        self.num_kernels = num_kernels
        
        # 创建多个卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            for _ in range(num_kernels)
        ])
        
        # 注意力网络，用于选择卷积核
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_kernels, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 生成动态权重
        attention_weights = self.attention(x)  # (B, num_kernels, 1, 1)
        
        # 计算所有卷积核的输出
        conv_outputs = [conv(x) for conv in self.convs]
        stacked = torch.stack(conv_outputs, dim=1)  # (B, num_kernels, C, H, W)
        
        # 应用注意力权重
        attention_weights = attention_weights.unsqueeze(-1)  # (B, num_kernels, 1, 1, 1)
        output = (stacked * attention_weights).sum(dim=1)  # (B, C, H, W)
        
        return output


class LSConv(nn.Module):
    """
    LS卷积模块：See Large + Focus Small
    
    结构:
    1. 大核深度卷积（See Large）：捕获全局上下文
    2. 小核动态卷积（Focus Small）：聚焦局部细节（眼睛/嘴巴/眉毛）
    3. 特征融合
    """
    
    def __init__(self, 
                 channels: int,
                 large_kernel: int = 21,
                 small_kernel: int = 3,
                 reduction: int = 4):
        """
        Args:
            channels: 通道数
            large_kernel: 大核尺寸（用于全局感知）
            small_kernel: 小核尺寸（用于局部聚焦）
            reduction: 特征压缩比例
        """
        super().__init__()
        
        # See Large: 大核深度卷积捕获全局结构
        self.see_large = DepthWiseConv(
            channels, large_kernel, 
            padding=large_kernel // 2
        )
        
        # Focus Small: 小核动态卷积聚焦局部细节
        self.focus_small = DynamicConv2d(
            channels, channels, 
            small_kernel, 
            num_kernels=4
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
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
        
        # 通道注意力重标定
        ca = self.channel_attention(fused)
        fused = fused * ca
        
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
