"""
Kimi自注意力残差连接模块
替换标准残差连接 x = x + f(x)
通过Query向量引导、Value向量辅助的方式重构残差融合
缓解深层Transformer的训练退化问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入 (B, N, C)
            mask: 注意力掩码 (B, N, N)
            
        Returns:
            out: 输出 (B, N, C)
            attn: 注意力权重 (B, num_heads, N, N)
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class SelfAttentionResidual(nn.Module):
    """
    Kimi自注意力残差连接
    
    核心思想:
    1. 使用Query向量引导残差融合的方向
    2. 使用Value向量辅助特征增强
    3. 动态调整残差权重，避免特征冗余
    
    公式:
        output = alpha * x + (1 - alpha) * (attn_output + value_guided)
    其中 alpha 由 Query 自适应生成
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Query引导的权重生成器
        self.query_guide = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Value辅助特征生成器
        self.value_assist = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 跨层信息聚合
        self.cross_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x, query, value, attn_output):
        """
        Args:
            x: 输入特征 (B, N, C)
            query: Query向量 (B, N, C)
            value: Value向量 (B, N, C)
            attn_output: 注意力输出 (B, N, C)
            
        Returns:
            output: 自注意力残差输出 (B, N, C)
            alpha: 残差权重 (B, N, 1)
        """
        # Query引导生成自适应权重
        alpha = self.query_guide(query)  # (B, N, 1)
        
        # Value辅助特征增强
        value_assisted = self.value_assist(value)  # (B, N, C)
        
        # 跨层信息聚合
        cross_info = self.cross_layer(
            torch.cat([x, attn_output], dim=-1)
        )  # (B, N, C)
        
        # 自注意力残差融合
        # alpha控制原始特征的保留程度
        # (1-alpha)控制注意力特征和新特征的影响
        output = alpha * x + (1 - alpha) * (attn_output + value_assisted + cross_info)
        
        return output, alpha


class KimiAttentionBlock(nn.Module):
    """
    Kimi自注意力块
    包含：MHSA + 自注意力残差连接
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        
        # 多头自注意力
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 自注意力残差连接
        self.residual = SelfAttentionResidual(embed_dim)
        
        # MLP (FFN)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_attn=False):
        """
        Args:
            x: 输入 (B, N, C)
            return_attn: 是否返回注意力权重
            
        Returns:
            output: 输出 (B, N, C)
            attn_weights: 注意力权重 (可选)
        """
        identity = x
        
        # 层归一化
        x_norm = self.norm1(x)
        
        # 多头自注意力
        attn_output, attn_weights = self.attn(x_norm)
        
        # 自注意力残差连接（Kimi）
        # Query和Value使用归一化后的输入
        # qkv = self.attn.qkv(x_norm)
        # q = qkv[..., :self.attn.embed_dim]
        # v = qkv[..., 2*self.attn.embed_dim:]
        # 为了简化，直接使用x_norm作为query和value
        x, alpha = self.residual(identity, x_norm, x_norm, attn_output)
        
        # 第二个残差（标准残差用于MLP）
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        if return_attn:
            return x, attn_weights, alpha
        return x


class StandardResidual(nn.Module):
    """标准残差连接（用于对比实验）"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, attn_output):
        return x + attn_output


class StandardAttentionBlock(nn.Module):
    """标准注意力块（用于对比实验）"""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.residual = StandardResidual()
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_attn=False):
        identity = x
        x = self.norm1(x)
        attn_output, attn_weights = self.attn(x)
        x = self.residual(identity, attn_output)
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        if return_attn:
            return x, attn_weights
        return x


# 测试代码
if __name__ == "__main__":
    print("测试Kimi自注意力残差连接模块...")
    
    # 测试SelfAttentionResidual
    x = torch.randn(2, 196, 256)
    query = torch.randn(2, 196, 256)
    value = torch.randn(2, 196, 256)
    attn_output = torch.randn(2, 196, 256)
    
    residual = SelfAttentionResidual(256)
    output, alpha = residual(x, query, value, attn_output)
    print(f"SelfAttentionResidual: 输入{x.shape}, 输出{output.shape}, alpha{alpha.shape}")
    print(f"alpha范围: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    
    # 测试KimiAttentionBlock
    block = KimiAttentionBlock(embed_dim=256, num_heads=8)
    x = torch.randn(2, 196, 256)
    output, attn_weights, alpha = block(x, return_attn=True)
    print(f"\nKimiAttentionBlock: 输入{(2, 196, 256)}, 输出{output.shape}")
    print(f"注意力权重: {attn_weights.shape}, 残差权重: {alpha.shape}")
    
    # 测试StandardAttentionBlock（对比）
    std_block = StandardAttentionBlock(embed_dim=256, num_heads=8)
    std_output, std_attn = std_block(x, return_attn=True)
    print(f"\nStandardAttentionBlock: 输出{std_output.shape}")
    
    # 参数量对比
    kimi_params = sum(p.numel() for p in block.parameters())
    std_params = sum(p.numel() for p in std_block.parameters())
    print(f"\n参数量对比:")
    print(f"Kimi: {kimi_params:,}")
    print(f"Standard: {std_params:,}")
    print(f"差异: {kimi_params - std_params:,} ({(kimi_params/std_params-1)*100:.2f}%)")
    
    print("\nKimi自注意力残差连接模块测试通过!")
