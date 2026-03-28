"""
ViT-LSNet 训练脚本

优化内容:
1. ✅ 移除TensorBoard，改用CSV日志
2. ✅ 保存最优模型的注意力图、混淆矩阵、热力图
3. ✅ 添加学习率衰减（ReduceLROnPlateau）
4. ✅ 保存每个epoch的test和train的loss、accuracy
5. ✅ 监控增强：过拟合指标、梯度范数、类别准确率分析
6. ✅ 配置解耦：支持命令行参数和配置文件
"""

# 标准库导入
import os
import sys
import json
import time
import csv
from typing import Dict, Optional

# 第三方库导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 项目模块导入
from vit_lsnet_fer import ViTLSNetFER, ViTLSNetFERConfig, EMOTION_LABELS

# 数据集模块导入
import dataset as dataset_module

SyntheticFERDataset = dataset_module.SyntheticFERDataset
create_fer2013_dataloaders = getattr(dataset_module, 'create_fer2013_dataloaders', None)

# Focal Loss导入
try:
    from focal_loss import FocalLoss
except ImportError:
    FocalLoss = None

# 配置导入
try:
    from train_config import get_config, save_config
except ImportError:
    # 如果train_config不存在，使用默认配置
    import argparse
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=120)
        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--save_dir', type=str, default='./checkpoints')
        parser.add_argument('--log_dir', type=str, default='./logs')
        args = parser.parse_args()
        # 添加默认属性
        args.weight_decay = 1e-3
        args.early_stopping_patience = 8
        args.use_mixed_precision = True
        args.save_attention = True
        args.save_confusion_matrix = True
        args.use_focal_loss = True
        args.focal_gamma = 1.5
        args.use_ls_conv = True
        args.use_kimi_residual = True
        args.lr_scheduler = 'reduce_on_plateau'
        args.lr_patience = 5
        args.lr_factor = 0.5
        args.lr_min = 1e-6
        args.grad_clip = 1.0
        args.use_class_weights = True
        args.class_weights = [1.2, 8.0, 1.5, 0.6, 0.9, 1.3, 1.1]
        args.monitor_overfitting = True
        args.monitor_grad_norm = True
        args.monitor_class_accuracy = True
        args.save_period = 10
        args.num_layers = 12
        args.embed_dim = 384
        args.num_heads = 6
        args.mlp_ratio = 4.0
        args.dropout = 0.1
        args.data_csv = 'FER2013.csv'
        args.num_workers = 4
        args.img_size = 224
        args.use_augmentation = True
        args.augmentation_prob = 0.5
        args.use_mtcnn = False
        return args

    def save_config(args, save_path='./checkpoints/config.json'):
        pass


def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f'使用GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU数量: {torch.cuda.device_count()}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB')
    else:
        device = 'cpu'
        print('使用CPU')

    return device


class Trainer:
    """训练器 - 优化版"""

    def __init__(self, model, train_loader, test_loader, args):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            args: 配置参数（从train_config.get_config()获取）
        """
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.save_dir = args.save_dir
        self.log_dir = args.log_dir
        self.use_mixed_precision = args.use_mixed_precision
        self.save_attention = args.save_attention
        self.save_confusion_matrix = args.save_confusion_matrix
        self.grad_clip = args.grad_clip
        self.monitor_overfitting = args.monitor_overfitting
        self.monitor_grad_norm = args.monitor_grad_norm
        self.monitor_class_accuracy = args.monitor_class_accuracy
        self.save_period = args.save_period

        # 消融实验模式
        self.ablation_mode = {
            'use_ls_conv': args.use_ls_conv,
            'use_kimi_residual': args.use_kimi_residual
        }

        # 创建目录
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'attention_maps'), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'confusion_matrices'), exist_ok=True)

        # 损失函数
        if args.use_focal_loss and FocalLoss is not None:
            class_weights = torch.tensor(args.class_weights).to(self.device) if args.use_class_weights else None
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=args.focal_gamma
            )
            print('使用Focal Loss')
        else:
            class_weights = torch.tensor(args.class_weights).to(self.device) if args.use_class_weights else None
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights
            )
            print('使用CrossEntropyLoss')

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 学习率调度器
        if args.lr_scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.lr_min,
                verbose=True
            )
        elif args.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.num_epochs,
                eta_min=args.lr_min
            )
        else:  # step
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.num_epochs // 3,
                gamma=0.1
            )

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if args.use_mixed_precision else None

        # 训练状态
        self.current_epoch = 0
        self.best_acc = 0.0
        self.early_stop_count = 0
        self.early_stopping_patience = args.early_stopping_patience
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.learning_rates = []
        self.epoch_times = []
        self.overfitting_gaps = []
        self.grad_norms = []
        self.class_accuracies = []

        # CSV日志文件
        self.training_log_path = os.path.join(args.log_dir, 'training_log.csv')
        self._init_training_log()

        print(f'初始化完成')
        print(f'  学习率调度器: {args.lr_scheduler}')
        print(f'  早停patience: {args.early_stopping_patience}')
        print(f'  梯度裁剪: {args.grad_clip if args.grad_clip > 0 else "关闭"}')

    def _init_training_log(self):
        """初始化训练日志CSV"""
        with open(self.training_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                'Epoch',
                'Train_Loss',
                'Train_Accuracy(%)',
                'Test_Loss',
                'Test_Accuracy(%)',
                'Overfitting_Gap(%)',
                'Learning_Rate',
                'Epoch_Time(s)',
                'Grad_Norm',
                'No_Improve_Count',
                'Best_Test_Acc(%)'
            ])
        print(f'训练日志已初始化: {self.training_log_path}')

    def _log_epoch_to_csv(self, train_loss, train_acc, test_loss, test_acc,
                        epoch_time, lr, grad_norm):
        """记录epoch到CSV日志"""
        overfitting_gap = train_acc - test_acc
        self.overfitting_gaps.append(overfitting_gap)

        with open(self.training_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_epoch + 1,
                f'{train_loss:.6f}',
                f'{train_acc:.4f}',
                f'{test_loss:.6f}',
                f'{test_acc:.4f}',
                f'{overfitting_gap:.4f}',
                f'{lr:.8f}',
                f'{epoch_time:.2f}',
                f'{grad_norm:.6f}' if grad_norm > 0 else 'N/A',
                self.early_stop_count,
                f'{self.best_acc:.4f}'
            ])

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs}',
                    leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_loss = total_loss / len(self.train_loader)
        acc = 100. * correct / total

        self.train_losses.append(avg_loss)
        self.train_accs.append(acc)

        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)

        return avg_loss, acc, epoch_time

    @torch.no_grad()
    def test(self, save_cm=False):
        """测试模型"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # 类别准确率统计
        class_correct = [0] * 7
        class_total = [0] * 7

        pbar = tqdm(self.test_loader, desc='Testing', leave=False)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 统计每个类别的准确率
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.test_loader)
        acc = 100. * correct / total

        self.test_losses.append(avg_loss)
        self.test_accs.append(acc)

        # 计算类别准确率
        class_acc = []
        for i in range(7):
            if class_total[i] > 0:
                class_acc.append(100. * class_correct[i] / class_total[i])
            else:
                class_acc.append(0.0)
        self.class_accuracies.append(class_acc)

        # 保存混淆矩阵
        if save_cm and len(all_preds) > 0:
            self._save_confusion_matrix(all_labels, all_preds)

        return avg_loss, acc

    def _save_confusion_matrix(self, y_true, y_pred):
        """保存混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        # 使用seaborn绘制
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(EMOTION_LABELS.values()),
                    yticklabels=list(EMOTION_LABELS.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch + 1}')

        # 保存
        save_path = os.path.join(self.log_dir, 'confusion_matrices',
                                 f'confusion_matrix_epoch_{self.current_epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'混淆矩阵已保存: {save_path}')

    @torch.no_grad()
    def save_attention_maps(self, num_samples=4):
        """保存注意力热力图"""
        print('\n生成注意力热力图...')

        self.model.eval()

        sample_count = 0
        for images, labels in self.test_loader:
            if sample_count >= num_samples:
                break

            images = images[:min(4, len(images))].to(self.device)

            # 获取注意力图
            if hasattr(self.model, 'get_attention_maps'):
                attention_maps = self.model.get_attention_maps(images, layer_idx=-1)

                # 保存每个样本
                for i in range(min(4, len(images))):
                    attn_map = attention_maps[i].cpu().numpy()
                    attn_map = attn_map.mean(0)  # 平均多头

                    # 归一化
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                    # 使用matplotlib绘制
                    plt.figure(figsize=(8, 8))
                    plt.imshow(attn_map, cmap='hot')
                    plt.colorbar()
                    plt.title(f'Attention Map - {EMOTION_LABELS[labels[i].item()]}')

                    save_path = os.path.join(self.log_dir, 'attention_maps',
                                             f'attention_ep{self.current_epoch + 1}_sample{sample_count + i + 1}.png')
                    plt.savefig(save_path, dpi=150)
                    plt.close()

            sample_count += len(images)

        print(f'已保存 {num_samples} 个注意力热力图')

    def _check_early_stopping(self, test_acc):
        """检查早停条件"""
        # 更新最佳准确率
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.early_stop_count = 0
            self.save_checkpoint(is_best=True)
            print(f'  ✓ 测试准确率提升至: {test_acc:.2f}% (保存最佳模型)')
        else:
            self.early_stop_count += 1
            print(f'  ⚠️ 测试准确率未改善 ({self.early_stop_count}/{self.early_stopping_patience})')

        # 早停检查
        if self.early_stop_count >= self.early_stopping_patience:
            print('\n' + '='*60)
            print('⚠️ 早停触发!')
            print('='*60)
            print(f'  连续 {self.early_stopping_patience} 个epoch准确率未改善')
            print(f'  最佳测试准确率: {self.best_acc:.2f}%')
            print(f'  在Epoch {self.current_epoch + 1}/{self.num_epochs} 停止训练')
            print('='*60)
            return True

        return False

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'test_accs': self.test_accs,
            'learning_rates': self.learning_rates,
            'ablation_mode': self.ablation_mode
        }

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
            print(f'  保存最佳模型: {path}')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.current_epoch + 1}.pth')

        torch.save(checkpoint, path)

    def train(self):
        """完整训练流程"""
        print('\n' + '='*60)
        print('训练配置')
        print('='*60)
        print(f'  设备: {self.device}')
        print(f'  模型参数量: {sum(p.numel() for p in self.model.parameters()):,}')
        print(f'  混合精度训练: {self.use_mixed_precision}')
        print(f'  保存注意力图: {self.save_attention}')
        print(f'  保存混淆矩阵: {self.save_confusion_matrix}')
        print(f'  监控过拟合: {self.monitor_overfitting}')
        print(f'  监控梯度范数: {self.monitor_grad_norm}')
        print(f'  监控类别准确率: {self.monitor_class_accuracy}')
        print(f'  消融实验: LS-Conv={self.ablation_mode["use_ls_conv"]}, Kimi-Residual={self.ablation_mode["use_kimi_residual"]}')
        print('='*60)

        print('\n开始训练...\n')

        start_time = time.time()

        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch

                print(f'\nEpoch [{epoch + 1}/{self.num_epochs}]')
                print('-' * 40)

                # 训练
                train_loss, train_acc, epoch_time = self.train_epoch()
                print(f'  训练Loss: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')

                # 测试
                test_loss, test_acc = self.test(save_cm=self.save_confusion_matrix)
                print(f'  测试Loss: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')

                # 计算梯度范数
                grad_norm = 0.0
                if self.monitor_grad_norm:
                    total_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2).item()
                            total_norm += param_norm ** 2
                    grad_norm = total_norm ** 0.5
                    self.grad_norms.append(grad_norm)
                    print(f'  梯度范数: {grad_norm:.6f}')

                # 更新学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.scheduler.step(test_acc)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                print(f'  学习率: {current_lr:.6f}')

                # 记录到CSV
                self._log_epoch_to_csv(train_loss, train_acc, test_loss, test_acc,
                                     epoch_time, current_lr, grad_norm)

                # 过拟合监控
                overfitting_gap = train_acc - test_acc
                if self.monitor_overfitting:
                    print(f'  过拟合差距: {overfitting_gap:.2f}%')
                    if overfitting_gap > 20.0:
                        print(f'  ⚠️ 警告: 过拟合严重!')

                # 类别准确率监控
                if self.monitor_class_accuracy and len(self.class_accuracies) > 0:
                    class_acc = self.class_accuracies[-1]
                    print(f'  类别准确率:', end='')
                    for i, acc in enumerate(class_acc):
                        print(f' {EMOTION_LABELS[i]}={acc:.1f}%', end='')
                    print()

                # 早停检查
                if self._check_early_stopping(test_acc):
                    break

                # 保存注意力热力图（每10个epoch保存一次）
                if self.save_attention and (epoch + 1) % 10 == 0:
                    # 传入模型输入图像而非attention maps
                    self.save_attention_maps(num_samples=4)

                # 定期保存模型
                if (epoch + 1) % self.save_period == 0:
                    self.save_checkpoint(is_best=False)
                    print(f'  定期保存检查点: epoch_{epoch + 1}')

                # ETA计算
                if epoch > 0:
                    avg_time = np.mean(self.epoch_times[:epoch])
                    eta = avg_time * (self.num_epochs - epoch - 1) / 60
                    print(f'  预计剩余时间: {eta:.1f} min')

        except KeyboardInterrupt:
            print('\n训练被中断')
            self.save_checkpoint(is_best=False)

        # 训练完成
        print('\n' + '='*60)
        print('训练完成!')
        print('='*60)
        print(f'  最佳测试准确率: {self.best_acc:.2f}%')
        print(f'  平均每个epoch时间: {np.mean(self.epoch_times):.2f}s')
        print(f'  总训练时间: {sum(self.epoch_times) / 60:.1f} min')

        # 过拟合分析
        if self.monitor_overfitting and len(self.overfitting_gaps) > 0:
            avg_gap = np.mean(self.overfitting_gaps)
            print(f'  平均过拟合差距: {avg_gap:.2f}%')

        print('='*60)

        # 保存最终混淆矩阵和分类报告
        if self.save_confusion_matrix:
            self._save_final_report()

        # 绘制训练曲线
        self._plot_training_curves()

    def _save_final_report(self):
        """保存最终混淆矩阵和分类报告"""
        print('\n保存最终混淆矩阵和分类报告...')

        # 获取所有预测
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 生成分类报告
        report = classification_report(all_labels, all_preds,
                                       target_names=list(EMOTION_LABELS.values()),
                                       output_dict=True)

        # 保存到文件
        report_path = os.path.join(self.save_dir, 'classification_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f'分类报告已保存到 {report_path}')

    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 训练准确率
        axes[0, 0].plot(range(1, len(self.train_accs) + 1), self.train_accs,
                     'b-', linewidth=2, label='训练准确率')
        axes[0, 0].plot(range(1, len(self.test_accs) + 1), self.test_accs,
                     'r-', linewidth=2, label='测试准确率')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('准确率 (%)')
        axes[0, 0].set_title('训练曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 训练损失
        axes[0, 1].plot(range(1, len(self.train_losses) + 1), self.train_losses,
                     'b-', linewidth=2, label='训练损失')
        axes[0, 1].plot(range(1, len(self.test_losses) + 1), self.test_losses,
                     'r-', linewidth=2, label='测试损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].set_title('损失曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 过拟合差距
        axes[1, 0].plot(range(1, len(self.overfitting_gaps) + 1), self.overfitting_gaps,
                     'g-', linewidth=2, label='过拟合差距')
        axes[1, 0].axhline(y=10.0, color='r', linestyle='--', alpha=0.5, label='警告线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('差距 (%)')
        axes[1, 0].set_title('过拟合监控')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 学习率
        axes[1, 1].plot(range(1, len(self.learning_rates) + 1), self.learning_rates,
                     'g-', linewidth=2, label='学习率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('学习率')
        axes[1, 1].set_title('学习率')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'训练曲线已保存: {save_path}')


def main():
    """主函数"""
    # 获取配置
    args = get_config()

    # 保存配置
    save_config(args)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = setup_device()

    # 检查数据集
    csv_file = args.data_csv
    use_grayscale = os.path.exists(csv_file)  # FER2013 是灰度图

    # 创建模型
    config = ViTLSNetFERConfig.LIGHT.copy()
    config['in_channels'] = 1 if use_grayscale else 3
    config['num_layers'] = args.num_layers
    config['embed_dim'] = args.embed_dim
    config['num_heads'] = args.num_heads
    config['mlp_ratio'] = args.mlp_ratio
    config['dropout'] = args.dropout
    config['ls_block_layers'] = getattr(args, 'ls_block_layers', 8)  # 使用命令行参数或默认8层

    model = ViTLSNetFER(**config)
    print('\n' + '='*60)
    print('模型配置')
    print('='*60)
    print(f'  输入通道: {1 if use_grayscale else 3} ({"灰度图" if use_grayscale else "彩色图"})')
    print(f'  嵌入维度: {model.embed_dim}')
    print(f'  Encoder层数: {model.encoder.num_layers}')
    print(f'  LS Block层数: {model.encoder.ls_block_layers}（{model.encoder.ls_block_layers/model.encoder.num_layers*100:.1f}%）')
    print(f'  MSA Block层数: {model.encoder.num_layers - model.encoder.ls_block_layers}（{(model.encoder.num_layers - model.encoder.ls_block_layers)/model.encoder.num_layers*100:.1f}%）')

    # 获取注意力头数（从使用MHSA的层获取）
    mhsa_layer_idx = model.encoder.ls_block_layers if model.encoder.ls_block_layers < model.encoder.num_layers else 0
    if mhsa_layer_idx < len(model.encoder.layers) and hasattr(model.encoder.layers[mhsa_layer_idx].attn_block, 'attn'):
        print(f'  注意力头数: {model.encoder.layers[mhsa_layer_idx].attn_block.attn.num_heads}')
    else:
        print(f'  注意力头数: {config.get("num_heads", args.num_heads)}')

    print(f'  使用LS卷积: {model.encoder.layers[0].use_ls_conv}')
    print(f'  LS Block层数: {model.encoder.ls_block_layers}（前{model.encoder.ls_block_layers}层无MHSA）')
    print(f'  MSA Block层数: {model.encoder.num_layers - model.encoder.ls_block_layers}')
    print(f'  使用Kimi残差: True')
    print(f'  模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    print('='*60)

    print('\n创建数据加载器...')

    if os.path.exists(csv_file):
        print(f'找到 {csv_file}，使用真实数据集')

        train_loader, test_loader = create_fer2013_dataloaders(
            csv_file=csv_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size
        )

        print(f'训练集: {len(train_loader.dataset)} 样本')
        print(f'测试集: {len(test_loader.dataset)} 样本')
    else:
        print(f'未找到 {csv_file}，使用合成数据集测试')
        print(f'提示: 将{csv_file}放在项目根目录以使用真实数据')

        train_dataset = SyntheticFERDataset(num_samples=1000, num_classes=7)
        test_dataset = SyntheticFERDataset(num_samples=200, num_classes=7)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f'训练集: {len(train_dataset)} 样本')
        print(f'测试集: {len(test_dataset)} 样本')
        print(f'表情类别: {EMOTION_LABELS}')

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        args=args
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
