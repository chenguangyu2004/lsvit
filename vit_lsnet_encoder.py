"""
ViT-LSNet融合的Transformer Encoder
核心创新：LS卷积 → MHSA → FFN 串行结构
严格遵循：先局部特征聚合，后全局信息交互
"""

import torch
import torch.nn as nn
from typing import List, Optional
import math

from ls_conv import LSConv
from self_attention_residual import KimiAttentionBlock, StandardAttentionBlock


class PatchEmbedding(nn.Module):
    """图像分块嵌入层"""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 384):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积进行分块嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            output: 嵌入后的序列 (B, N+1, D)
        """
        B = x.shape[0]
        
        # 分块嵌入 (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        return x


class SpatialToSequence(nn.Module):
    """空间特征转序列特征"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, H=None, W=None):
        """
        Args:
            x: 空间特征 (B, C, H, W)
            H: 目标高度（如果x已经展平，可以忽略）
            W: 目标宽度（如果x已经展平，可以忽略）
            
        Returns:
            output: 序列特征 (B, N+1, C), 包含CLS token
        """
        # 如果x是4D张量 (B, C, H, W)
        if x.dim() == 4:
            B, C, H, W = x.shape
            N = H * W
            
            # 展平为序列
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
            
            # 添加CLS token
            cls_token = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
            x = torch.cat([cls_token, x], dim=1)  # (B, N+1, C)
        
        # 如果x已经是3D张量 (B, N, C)，直接返回
        # 注意：这里假设CLS token已经存在
        
        return x


class SequenceToSpatial(nn.Module):
    """序列特征转空间特征"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, H, W):
        """
        Args:
            x: 序列特征 (B, N+1, C)
            H: 目标高度
            W: 目标宽度
            
        Returns:
            output: 空间特征 (B, C, H, W)
        """
        # 移除CLS token
        x = x[:, 1:, :]  # (B, N, C)
        
        # 转为空间维度
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        
        return x


class ViTLSNetEncoderLayer(nn.Module):
    """
    ViT-LSNet融合的Transformer Encoder层（支持可选MHSA）

    结构（严格顺序）:
    1. LS卷积：局部特征聚合（提取面部五官细节）
    2. 多头自注意力（MHSA）：全局信息交互（可选）
    3. 前馈网络（FFN）：特征变换

    使用Kimi自注意力残差连接替换标准残差连接
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 spatial_size: int = 14,  # 空间特征图尺寸
                 use_kimi_residual: bool = True,
                 use_ls_conv: bool = True,
                 use_mhsa: bool = True):  # 新增：是否使用MHSA
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            dropout: Dropout比例
            spatial_size: 空间特征图尺寸（用于LS卷积）
            use_kimi_residual: 是否使用Kimi自注意力残差连接
            use_ls_conv: 是否使用LS卷积
            use_mhsa: 是否使用多头自注意力
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.use_ls_conv = use_ls_conv
        self.use_mhsa = use_mhsa

        # 序列-空间转换
        self.seq2spatial = SequenceToSpatial()
        self.spatial2seq = SpatialToSequence()

        # 1. LS卷积：局部特征聚合
        if use_ls_conv:
            from ls_conv import LSConv
            self.ls_conv = LSConv(embed_dim)
        else:
            self.ls_conv = nn.Identity()

        # 2. 多头自注意力 + Kimi残差连接（可选）
        if use_mhsa:
            if use_kimi_residual:
                self.attn_block = KimiAttentionBlock(
                    embed_dim, num_heads, mlp_ratio, dropout
                )
            else:
                self.attn_block = StandardAttentionBlock(
                    embed_dim, num_heads, mlp_ratio, dropout
                )
        else:
            # 不使用MHSA时，只使用FFN
            self.attn_block = FFNBlock(embed_dim, mlp_ratio, dropout)

        # 3. 前馈网络（FFN）已经在attn_block中包含
        
    def forward(self, x, return_attn=False):
        """
        Args:
            x: 输入序列 (B, N+1, C)
            return_attn: 是否返回注意力权重

        Returns:
            output: 输出序列 (B, N+1, C)
            attn_weights: 注意力权重（可选）
            alpha: Kimi残差权重（可选）
        """
        identity = x

        # 转换为空间维度进行LS卷积
        spatial_x = self.seq2spatial(x, self.spatial_size, self.spatial_size)

        # 1. LS卷积：局部特征聚合
        # 提取面部五官细节特征（眼睛、嘴巴、眉毛）
        if self.use_ls_conv:
            spatial_x = self.ls_conv(spatial_x)

        # 转换回序列维度
        x = self.spatial2seq(spatial_x, self.spatial_size, self.spatial_size)

        # 2. 可选的多头自注意力：全局信息交互
        # 如果use_mhsa=False，只执行FFN
        if self.use_mhsa:
            if return_attn:
                x, attn_weights, alpha = self.attn_block(x, return_attn=True)
            else:
                x = self.attn_block(x)
                attn_weights = None
                alpha = None
        else:
            # 只执行FFN（LS Block）
            x = self.attn_block(x)
            attn_weights = None
            alpha = None

        if return_attn:
            return x, attn_weights, alpha
        return x


class ViTLSNetEncoder(nn.Module):
    """
    ViT-LSNet融合的Transformer Encoder（分层架构版）

    前几层：LS Block（LSConv + FFN，无MHSA）
    后几层：MSA Block（MHSA + FFN）
    """

    def __init__(self,
                 embed_dim: int = 384,
                 num_layers: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 spatial_size: int = 14,
                 use_kimi_residual: bool = True,
                 use_ls_conv: bool = True,
                 ls_block_layers: int = 4):  # 前几层只用LSConv
        """
        Args:
            embed_dim: 嵌入维度
            num_layers: Encoder层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            dropout: Dropout比例
            spatial_size: 空间特征图尺寸
            use_kimi_residual: 是否使用Kimi残差连接
            use_ls_conv: 是否使用LS卷积
            ls_block_layers: 前几层只用LSConv的层数
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ls_block_layers = ls_block_layers

        # 构建分层Encoder
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 前ls_block_layers层：LS Block（无MHSA）
            # 后面层：MSA Block（有MHSA）
            if i < ls_block_layers:
                layer = ViTLSNetEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    spatial_size=spatial_size,
                    use_kimi_residual=False,  # LS Block不需要Kimi残差
                    use_ls_conv=True,
                    use_mhsa=False  # 关键：不使用MHSA
                )
            else:
                layer = ViTLSNetEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    spatial_size=spatial_size,
                    use_kimi_residual=use_kimi_residual,
                    use_ls_conv=use_ls_conv,
                    use_mhsa=True  # 关键：使用MHSA
                )
            self.layers.append(layer)
        
    def forward(self, x, return_all_layers=False, return_attn=False):
        """
        Args:
            x: 输入序列 (B, N+1, C)
            return_all_layers: 是否返回所有层的输出
            return_attn: 是否返回注意力权重
            
        Returns:
            output: 输出序列 (B, N+1, C)
            all_outputs: 所有层输出（可选）
            attn_weights: 注意力权重（可选）
        """
        all_outputs = []
        all_attns = []
        all_alphas = []
        
        for layer in self.layers:
            if return_attn:
                x, attn_weights, alpha = layer(x, return_attn=True)
                all_attns.append(attn_weights)
                all_alphas.append(alpha)
            else:
                x = layer(x)
            
            if return_all_layers:
                all_outputs.append(x)
        
        if return_all_layers and return_attn:
            return x, all_outputs, all_attns, all_alphas
        elif return_all_layers:
            return x, all_outputs
        elif return_attn:
            return x, all_attns, all_alphas
        return x


class FFNBlock(nn.Module):
    """
    前馈网络块（FFN）
    用于LS Block中（无MHSA的情况）
    """

    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: 输入序列 (B, N+1, C)

        Returns:
            output: 输出序列 (B, N+1, C)
        """
        identity = x
        x = self.norm(x)
        x = self.mlp(x)
        x = x + identity
        return x


# 测试代码
if __name__ == "__main__":
    print("测试ViT-LSNet Encoder...")
    
    # 测试PatchEmbedding
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=384)
    x = torch.randn(2, 3, 224, 224)
    x = patch_embed(x)
    print(f"PatchEmbedding: 输入(2,3,224,224) -> 输出{x.shape}")
    
    # 测试ViTLSNetEncoderLayer
    encoder_layer = ViTLSNetEncoderLayer(
        embed_dim=384,
        num_heads=6,
        spatial_size=14,
        use_kimi_residual=True,
        use_ls_conv=True
    )
    output, attn, alpha = encoder_layer(x, return_attn=True)
    print(f"\nViTLSNetEncoderLayer: 输入{x.shape} -> 输出{output.shape}")
    print(f"注意力权重: {attn.shape}, Kimi残差权重: {alpha.shape}")
    
    # 测试ViTLSNetEncoder
    encoder = ViTLSNetEncoder(
        embed_dim=384,
        num_layers=12,
        num_heads=6,
        spatial_size=14,
        use_kimi_residual=True,
        use_ls_conv=True
    )
    output, all_attns, all_alphas = encoder(x, return_attn=True)
    print(f"\nViTLSNetEncoder (12层): 输入{x.shape} -> 输出{output.shape}")
    print(f"收集到 {len(all_attns)} 层注意力权重, {len(all_alphas)} 层Kimi权重")
    
    # 参数量统计
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 对比不使用LS卷积的情况
    encoder_no_ls = ViTLSNetEncoder(
        embed_dim=384,
        num_layers=12,
        num_heads=6,
        spatial_size=14,
        use_kimi_residual=True,
        use_ls_conv=False
    )
    params_no_ls = sum(p.numel() for p in encoder_no_ls.parameters())
    print(f"不使用LS卷积的参数量: {params_no_ls:,}")
    print(f"LS卷积增加参数量: {total_params - params_no_ls:,} ({(total_params/params_no_ls-1)*100:.2f}%)")
    
    print("\nViT-LSNet Encoder测试通过!")
