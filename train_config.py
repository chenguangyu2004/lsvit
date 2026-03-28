"""训练配置文件
支持命令行参数和配置文件两种方式修改超参数
"""

import argparse
import json
import os


def get_config():
    """获取训练配置"""
    parser = argparse.ArgumentParser(description='ViT-LSNet Training Configuration')

    # ===== 数据相关 =====
    parser.add_argument('--data_csv', type=str, default='FER2013.csv',
                        help='数据集CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图像尺寸')

    # ===== 模型相关 =====
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='嵌入维度')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Encoder层数')
    parser.add_argument('--ls_block_layers', type=int, default=8,
                        help='LS Block层数（前N层，仅使用LSConv，无MHSA）')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='注意力头数')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP扩展比例')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout率')

    # ===== 训练相关 =====
    parser.add_argument('--num_epochs', type=int, default=120,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='初始学习率（降低至1e-4以适应FER2013噪声）')
    parser.add_argument('--weight_decay', type=float, default=2e-3,
                        help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='早停patience')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                        help='是否使用混合精度训练')

    # ===== 损失函数相关 =====
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='是否使用Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                        help='Focal Loss的gamma参数')

    # ===== 消融实验开关 =====
    parser.add_argument('--use_ls_conv', action='store_true', default=True,
                        help='是否使用LS卷积')
    parser.add_argument('--use_kimi_residual', action='store_true', default=True,
                        help='是否使用Kimi自注意力残差')
    parser.add_argument('--use_mtcnn', action='store_true', default=False,
                        help='是否使用MTCNN预处理')

    # ===== 保存相关 =====
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--save_attention', action='store_true', default=True,
                        help='是否保存注意力热力图')
    parser.add_argument('--save_confusion_matrix', action='store_true', default=True,
                        help='是否保存混淆矩阵')
    parser.add_argument('--save_period', type=int, default=10,
                        help='定期保存模型间隔（epoch）')

    # ===== 学习率调度器相关 =====
    parser.add_argument('--lr_scheduler', type=str, default='reduce_on_plateau',
                        choices=['reduce_on_plateau', 'cosine', 'step'],
                        help='学习率调度器类型')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='ReduceLROnPlateau的patience')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau的factor')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='最小学习率')

    # ===== 数据增强 =====
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='是否使用数据增强')
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                        help='数据增强概率')

    # ===== 梯度裁剪 =====
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪阈值（0表示不裁剪）')

    # ===== 类别权重 =====
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='是否使用类别权重')
    parser.add_argument('--class_weights', type=float, nargs=7,
                        default=[1.2, 8.0, 1.5, 0.6, 0.9, 1.3, 1.1],
                        help='类别权重列表（7个类别）')

    # ===== 监控相关 =====
    parser.add_argument('--monitor_overfitting', action='store_true', default=True,
                        help='是否监控过拟合指标')
    parser.add_argument('--monitor_grad_norm', action='store_true', default=True,
                        help='是否监控梯度范数')
    parser.add_argument('--monitor_class_accuracy', action='store_true', default=True,
                        help='是否监控每个类别的准确率')

    # ===== 设备相关 =====
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='训练设备')

    args = parser.parse_args()

    # 如果存在config.json，则从文件加载并覆盖命令行参数
    config_file = 'config.json'
    if os.path.exists(config_file):
        print(f'从 {config_file} 加载配置...')
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        # 更新args
        for key, value in file_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def save_config(args, save_path='./checkpoints/config.json'):
    """保存配置到文件"""
    config_dict = vars(args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f'配置已保存到 {save_path}')
