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

        # 应用类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
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
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算softmax概率
        p = F.softmax(inputs, dim=1)

        # 获取真实类别的概率
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')

        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 计算focal loss
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


# 测试代码
if __name__ == "__main__":
    print("测试Focal Loss...")

    # 模拟输出和标签
    batch_size = 4
    num_classes = 7
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 3])

    # 标准交叉熵损失
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(inputs, targets)
    print(f"CrossEntropy Loss: {ce_value.item():.4f}")

    # Focal Loss (gamma=2.0)
    focal_loss = FocalLoss(gamma=2.0)
    focal_value = focal_loss(inputs, targets)
    print(f"Focal Loss (gamma=2.0): {focal_value.item():.4f}")

    # 带类别权重的Focal Loss
    class_weights = torch.tensor([1.07, 9.22, 1.00, 0.58, 0.82, 1.23, 0.83])
    weighted_focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    weighted_value = weighted_focal_loss(inputs, targets)
    print(f"Weighted Focal Loss: {weighted_value.item():.4f}")

    print("\nFocal Loss测试通过!")
