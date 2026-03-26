"""
完整的ViT-LSNet人脸表情识别模型

端到端架构:
1. MTCNN人脸检测与对齐
2. Patch Embedding
3. ViT-LSNet Encoder (LSConv → MHSA → FFN)
4. 分类头

支持灰度图和彩色图输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from mtcnn_detector import MTCNNDetector
from vit_lsnet_encoder import PatchEmbedding, ViTLSNetEncoder


class ViTLSNetFER(nn.Module):
    """
    ViT-LSNet人脸表情识别模型
    
    核心创新:
    1. MTCNN预处理：精准人脸检测与对齐
    2. LSConv-MHSA串行结构：先局部特征聚合，后全局信息交互
    3. Kimi自注意力残差连接：缓解深层网络退化
    """
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 7,
                 embed_dim: int = 384,
                 num_layers: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 spatial_size: int = 14,
                 use_kimi_residual: bool = True,
                 use_ls_conv: bool = True,
                 use_mtcnn: bool = False):
        """
        Args:
            img_size: 输入图像尺寸
            patch_size: 分块大小
            in_channels: 输入通道数（3为彩色，1为灰度）
            num_classes: 表情类别数（7类：生气、厌恶、恐惧、开心、悲伤、惊讶、中性）
            embed_dim: 嵌入维度
            num_layers: Encoder层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            dropout: Dropout比例
            spatial_size: 空间特征图尺寸
            use_kimi_residual: 是否使用Kimi自注意力残差连接
            use_ls_conv: 是否使用LS卷积
            use_mtcnn: 是否使用MTCNN预处理（推理时启用）
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_mtcnn = use_mtcnn
        
        # MTCNN人脸检测器（推理时使用）
        if use_mtcnn:
            self.mtcnn = MTCNNDetector(target_size=(img_size, img_size))
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # ViT-LSNet Encoder
        self.encoder = ViTLSNetEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            spatial_size=spatial_size,
            use_kimi_residual=use_kimi_residual,
            use_ls_conv=use_ls_conv
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x, return_attn=False, return_all_layers=False):
        """
        特征提取
        
        Args:
            x: 输入图像 (B, C, H, W)
            return_attn: 是否返回注意力权重
            return_all_layers: 是否返回所有层输出
            
        Returns:
            cls_token: CLS token特征
            attn_weights: 注意力权重（可选）
            all_outputs: 所有层输出（可选）
        """
        # Patch Embedding
        x = self.patch_embed(x)
        
        # ViT-LSNet Encoder
        if return_all_layers and return_attn:
            x, all_outputs, all_attns, all_alphas = self.encoder(
                x, return_all_layers=True, return_attn=True
            )
            cls_token = x[:, 0]
            return cls_token, all_outputs, all_attns, all_alphas
        elif return_attn:
            x, all_attns, all_alphas = self.encoder(x, return_attn=True)
            cls_token = x[:, 0]
            return cls_token, all_attns, all_alphas
        elif return_all_layers:
            x, all_outputs = self.encoder(x, return_all_layers=True)
            cls_token = x[:, 0]
            return cls_token, all_outputs
        else:
            x = self.encoder(x)
            cls_token = x[:, 0]
            return cls_token
    
    def forward(self, x, return_attn=False, return_all_layers=False):
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W) 或 图像列表（如果使用MTCNN）
            return_attn: 是否返回注意力权重
            return_all_layers: 是否返回所有层输出
            
        Returns:
            logits: 分类logits (B, num_classes)
            attn_weights: 注意力权重（可选）
        """
        # MTCNN预处理（仅推理时使用）
        if self.use_mtcnn and self.training == False:
            if isinstance(x, list):
                # 如果是图像列表，批量预处理
                tensors = []
                for img in x:
                    if isinstance(img, np.ndarray):
                        tensor = self.mtcnn.preprocess(img)
                        tensors.append(tensor)
                    else:
                        tensors.append(img)
                x = torch.stack(tensors, dim=0)
        
        # 特征提取
        if return_all_layers and return_attn:
            cls_token, all_outputs, all_attns, all_alphas = self.forward_features(
                x, return_attn=True, return_all_layers=True
            )
        elif return_attn:
            cls_token, all_attns, all_alphas = self.forward_features(
                x, return_attn=True
            )
        elif return_all_layers:
            cls_token, all_outputs = self.forward_features(
                x, return_all_layers=True
            )
        else:
            cls_token = self.forward_features(x)
        
        # 分类
        logits = self.classifier(cls_token)
        
        if return_all_layers and return_attn:
            return logits, all_outputs, all_attns, all_alphas
        elif return_attn:
            return logits, all_attns, all_alphas
        elif return_all_layers:
            return logits, all_outputs
        else:
            return logits
    
    def predict(self, x):
        """
        预测表情类别
        
        Args:
            x: 输入图像或图像列表
            
        Returns:
            pred_class: 预测类别
            pred_prob: 预测概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            pred_prob = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(pred_prob, dim=-1)
        return pred_class, pred_prob
    
    def get_attention_maps(self, x, layer_idx=-1):
        """
        获取注意力图用于可视化
        
        Args:
            x: 输入图像
            layer_idx: 层索引（-1表示最后一层）
            
        Returns:
            attention_map: 注意力图 (B, num_heads, N, N)
        """
        self.eval()
        with torch.no_grad():
            _, all_attns, _ = self.forward(x, return_attn=True)
            attention_map = all_attns[layer_idx]
        return attention_map


class ViTLSNetFERConfig:
    """模型配置类"""
    
    # 轻量级配置（适合RTX4060）
    LIGHT = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 7,
        'embed_dim': 384,
        'num_layers': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'spatial_size': 14,
        'use_kimi_residual': True,
        'use_ls_conv': True,
    }
    
    # 超轻量级配置
    TINY = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 7,
        'embed_dim': 192,
        'num_layers': 8,
        'num_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'spatial_size': 14,
        'use_kimi_residual': True,
        'use_ls_conv': True,
    }
    
    # 标准配置
    BASE = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 7,
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'spatial_size': 14,
        'use_kimi_residual': True,
        'use_ls_conv': True,
    }


# 表情类别标签
EMOTION_LABELS = {
    0: 'Angry',      # 生气
    1: 'Disgust',    # 厌恶
    2: 'Fear',       # 恐惧
    3: 'Happy',      # 开心
    4: 'Sad',        # 悲伤
    5: 'Surprise',   # 惊讶
    6: 'Neutral'     # 中性
}


# 测试代码
if __name__ == "__main__":
    print("测试ViT-LSNet人脸表情识别模型...")
    
    # 创建模型（轻量级配置）
    model = ViTLSNetFER(**ViTLSNetFERConfig.LIGHT)
    model.eval()
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"\n输入: {x.shape}")
    print(f"输出logits: {logits.shape}")
    
    # 测试预测
    pred_class, pred_prob = model.predict(x)
    print(f"预测类别: {pred_class}")
    print(f"预测概率: {pred_prob}")
    
    # 测试获取注意力图
    attn_map = model.get_attention_maps(x)
    print(f"\n注意力图形状: {attn_map.shape}")
    
    # 测试返回所有层输出
    logits, all_outputs, all_attns, all_alphas = model(
        x, return_attn=True, return_all_layers=True
    )
    print(f"所有层输出数量: {len(all_outputs)}")
    print(f"所有注意力权重数量: {len(all_attns)}")
    print(f"所有Kimi残差权重数量: {len(all_alphas)}")
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试不同配置
    print("\n不同配置对比:")
    for config_name, config in [('TINY', ViTLSNetFERConfig.TINY),
                                   ('LIGHT', ViTLSNetFERConfig.LIGHT),
                                   ('BASE', ViTLSNetFERConfig.BASE)]:
        model = ViTLSNetFER(**config)
        params = sum(p.numel() for p in model.parameters())
        print(f"{config_name}: {params:,} 参数")
    
    # 测试灰度图输入
    print("\n测试灰度图输入:")
    model_gray = ViTLSNetFER(in_channels=1, **ViTLSNetFERConfig.LIGHT)
    x_gray = torch.randn(2, 1, 224, 224)
    logits_gray = model_gray(x_gray)
    print(f"灰度图输入: {x_gray.shape} -> 输出: {logits_gray.shape}")
    
    print("\nViT-LSNet人脸表情识别模型测试通过!")
    print(f"\n表情类别: {EMOTION_LABELS}")
