"""
Focal Loss实现
用于解决类别不平衡和难分类样本问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: 用于解决类别不平衡

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha (float or tensor): 类别权重
        gamma (float): 聚焦参数，γ越大越关注难分类样本
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出 (B, C)，未经过softmax
            targets: 真实标签 (B,)

        Returns:
            loss
        """
        # 计算softmax概率
        p = F.softmax(inputs, dim=1)

        # 获取真实类别的概率
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 应用类别权重（修复设备问题）
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # 确保alpha在正确的设备上
                alpha = self.alpha.to(inputs.device) if self.alpha.device != inputs.device else self.alpha
                alpha_t = alpha[targets]

            focal_weight = alpha_t * focal_weight

        # 计算focal loss
        focal_loss = focal_weight * ce_loss

        # reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    加权Focal Loss: 结合类别权重和focal loss

    Args:
        class_weights: 类别权重tensor
        gamma: 聚焦参数
    """

    def __init__(self, class_weights=None, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()

        if class_weights is not None:
            # 转换为tensor并设置设备（将在forward中使用）
            self.register_buffer('alpha', class_weights.float())
        else:
            self.alpha = None

        self.gamma = gamma

    def forward(self, inputs, targets):
        # 使用FocalLoss
        focal_loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        return focal_loss_fn(inputs, targets)
