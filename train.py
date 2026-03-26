

# 标准库导入
import os
import sys
import json
import time
from typing import Dict, Optional
from datetime import datetime

# 第三方库导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
    """训练器 - 增强版（详细记录每个batch）"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 num_epochs: int = 100,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 0.05,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 use_mixed_precision: bool = True,
                 save_attention: bool = True,
                 save_confusion_matrix: bool = True,
                 ablation_mode: Optional[Dict] = None,
                 log_batch_every: int = 10):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 设备
            save_dir: 模型保存目录
            log_dir: 日志保存目录
            use_mixed_precision: 是否使用混合精度训练
            save_attention: 是否保存注意力热力图
            save_confusion_matrix: 是否保存混淆矩阵
            ablation_mode: 消融实验模式
            log_batch_every: 每隔多少个batch记录一次详细信息
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.use_mixed_precision = use_mixed_precision
        self.save_attention = save_attention
        self.save_confusion_matrix = save_confusion_matrix
        self.ablation_mode = ablation_mode or {'use_ls_conv': True, 'use_kimi_residual': True}
        self.log_batch_every = log_batch_every

        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'attention_maps'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'confusion_matrices'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'batch_logs'), exist_ok=True)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.learning_rates = []
        self.epoch_times = []

        # 记录每个batch的详细信息
        self.batch_losses = []
        self.batch_accs = []

        print(f'初始化完成，每隔 {log_batch_every} 个batch记录详细信息')

    def log_batch_details(self, batch_idx, images, labels, outputs, loss, predicted, epoch_start_time):
        """记录每个batch的详细信息到TensorBoard"""
        global_step = self.current_epoch * len(self.train_loader) + batch_idx

        # 基础统计
        batch_acc = (predicted.eq(labels).sum().item() / labels.size(0) * 100)

        # 记录每个batch的loss和accuracy
        self.writer.add_scalar('Batch/train_loss', loss.item(), global_step)
        self.writer.add_scalar('Batch/train_accuracy', batch_acc, global_step)

        # 每隔log_batch_every个batch记录一次详细信息
        if batch_idx % self.log_batch_every == 0:
            # 统计每个类别的预测数量
            pred_distribution = torch.bincount(predicted, minlength=7).float()
            pred_distribution = pred_distribution / pred_distribution.sum() * 100

            # 记录每个类别的准确率
            class_acc = []
            for class_id in range(7):
                class_mask = (labels == class_id)
                if class_mask.sum() > 0:
                    class_correct = (predicted[class_mask] == labels[class_mask]).sum().item()
                    class_acc.append(class_correct / class_mask.sum().item() * 100)
                else:
                    class_acc.append(0.0)

            # 写入TensorBoard
            for i in range(7):
                self.writer.add_scalar(f'Batch/Class_{EMOTION_LABELS[i]}_Accuracy',
                                       class_acc[i], global_step)
                self.writer.add_scalar(f'Batch/Class_{EMOTION_LABELS[i]}_Pred_Rate',
                                       pred_distribution[i].item(), global_step)

        # ========== 【BUG修复 1】直方图只在有有效值时写入 ==========
        if batch_idx % 200 == 0:
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad and param.grad is not None:
                    if param.grad.numel() > 0:
                        self.writer.add_histogram(f'Weights/{name}', param.data, global_step)
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, global_step)

        # 每50个batch记录一次梯度范数
        if batch_idx % 50 == 0:
            total_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar('Batch/Gradient_Norm', total_norm, global_step)

        # 保存到列表
        self.batch_losses.append(loss.item())
        self.batch_accs.append(batch_acc)

    def train_epoch(self):
        """训练一个epoch - 详细记录每个batch"""
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
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 记录每个batch的详细信息
            self.log_batch_details(batch_idx, images, labels, outputs, loss, predicted, epoch_start_time)

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

        # 记录epoch级别的指标
        self.writer.add_scalar('Epoch/train_loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('Epoch/train_accuracy', acc, self.current_epoch)
        if len(self.batch_losses) > 0:
            self.writer.add_histogram('Epoch/train_loss_distribution', np.array(self.batch_losses), self.current_epoch)
            self.writer.add_histogram('Epoch/train_acc_distribution', np.array(self.batch_accs), self.current_epoch)

        # 重置batch记录
        self.batch_losses = []
        self.batch_accs = []

        return avg_loss, acc

    @torch.no_grad()
    def test(self, save_cm=False):
        """测试模型"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        test_losses = []

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
            test_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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

        # 记录到TensorBoard
        self.writer.add_scalar('Epoch/test_loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('Epoch/test_accuracy', acc, self.current_epoch)
        if len(test_losses) > 0:
            self.writer.add_histogram('Epoch/test_loss_distribution', np.array(test_losses), self.current_epoch)

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

        # 保存到 log_dir 下的 confusion_matrices 目录
        save_path = os.path.join(self.log_dir, 'confusion_matrices',
                                 f'confusion_matrix_epoch_{self.current_epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 保存到TensorBoard
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(EMOTION_LABELS.values()),
                    yticklabels=list(EMOTION_LABELS.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch + 1}')
        self.writer.add_figure('Confusion_Matrix', fig, self.current_epoch)
        plt.close()

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

    def train(self):
        """完整训练流程"""
        print('\n' + '=' * 60)
        print('训练配置')
        print('=' * 60)
        print(f'  设备: {self.device}')
        print(f'  模型参数量: {sum(p.numel() for p in self.model.parameters()):,}')
        print(f'  混合精度训练: {self.use_mixed_precision}')
        print(f'  保存注意力图: {self.save_attention}')
        print(f'  保存混淆矩阵: {self.save_confusion_matrix}')
        print(
            f'  消融实验: LS-Conv={self.ablation_mode["use_ls_conv"]}, Kimi-Residual={self.ablation_mode["use_kimi_residual"]}')
        print(f'  Batch详细记录: 每隔 {self.log_batch_every} 个batch')
        print('=' * 60)

        # 记录消融实验配置
        self.writer.add_text('Ablation/Config',
                             json.dumps(self.ablation_mode, indent=2),
                             0)

        # 记录超参数
        hparam_dict = {
            'num_epochs': self.num_epochs,
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            'weight_decay': self.optimizer.param_groups[0]["weight_decay"],
            'batch_size': self.train_loader.batch_size,
            'use_mixed_precision': self.use_mixed_precision,
            'use_ls_conv': self.ablation_mode['use_ls_conv'],
            'use_kimi_residual': self.ablation_mode['use_kimi_residual']
        }
        self.writer.add_hparams(hparam_dict, {'hparam/accuracy': 0})

        print('\n开始训练...\n')

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 测试
            test_loss, test_acc = self.test(save_cm=self.save_confusion_matrix)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            self.learning_rates.append(current_lr)

            # 记录到TensorBoard
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Epoch/Time', epoch_time, epoch)

            # 打印结果
            print(f'\nEpoch [{epoch + 1}/{self.num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            print(f'Epoch Time: {epoch_time:.2f}s')
            if epoch > 0:
                avg_time = np.mean(self.epoch_times[:epoch])
                eta = avg_time * (self.num_epochs - epoch - 1) / 60
                print(f'ETA: {eta:.1f} min')

            # 保存注意力热力图（每10个epoch保存一次）
            if self.save_attention and (epoch + 1) % 10 == 0:
                self.save_attention_maps(num_samples=4)

            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f'保存最佳模型 (测试准确率: {test_acc:.2f}%)')

            # 定期保存模型
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        print('\n' + '=' * 60)
        print('训练完成!')
        print('=' * 60)
        print(f'  最佳测试准确率: {self.best_acc:.2f}%')
        print(f'  平均每个epoch时间: {np.mean(self.epoch_times):.2f}s')
        print(f'  总训练时间: {sum(self.epoch_times) / 60:.1f} min')
        print('=' * 60)

        # 保存最终混淆矩阵和分类报告
        if self.save_confusion_matrix:
            print('\n保存最终混淆矩阵和分类报告...')
            self._save_confusion_matrix_and_report()

        self.writer.close()
        print('\nTensorBoard日志已保存到 ./logs')
        print('运行以下命令查看：tensorboard --logdir=./logs')

    def _save_confusion_matrix_and_report(self):
        """保存最终混淆矩阵和分类报告"""
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
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f'分类报告已保存到 {report_path}')

        # 记录到TensorBoard
        self.writer.add_text('Final/Classification_Report',
                             json.dumps(report, indent=2),
                             self.num_epochs - 1)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
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
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')

        torch.save(checkpoint, path)

        # 同时保存模型配置
        config = {
            'best_acc': self.best_acc,
            'num_epochs': self.num_epochs,
            'train_losses': self.train_losses,
            'test_accs': self.test_accs,
            'learning_rates': self.learning_rates,
            'ablation_mode': self.ablation_mode
        }
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = setup_device()
    
    # 检查数据集格式并自动判断是否为灰度图
    csv_file = 'FER2013.csv'
    use_grayscale = False  # 默认值
    
    # 使用 try-except 捕获数据集加载异常，并提供回退方案
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # 检查数据集格式，自动判断是否为灰度图（48x48 = 2304 像素）
        if 'pixels' in df.columns and 'emotion' in df.columns:
            sample_pixels = df.iloc[0]['pixels'].split()
            is_grayscale = len(sample_pixels) == 2304  # 48x48 = 2304 像素
            print(f'✓ 检测到 {csv_file}，确认为灰度图数据集')
            use_grayscale = is_grayscale
        else:
            print(f'⚠ {csv_file} 格式非标准，使用默认设置')
    
    except FileNotFoundError:
        print(f'⚠ 未找到 {csv_file}，将使用合成数据集')
    except Exception as e:
        print(f'⚠ 数据集检查失败: {str(e)}，将使用合成数据集')

    # 检查是否有真实数据集（FER2013.csv）
    csv_file = 'FER2013.csv'
    use_grayscale = os.path.exists(csv_file)  # FER2013 是灰度图

    # 创建模型
    config = ViTLSNetFERConfig.LIGHT.copy()
    config['in_channels'] = 1 if use_grayscale else 3

    model = ViTLSNetFER(**config)
    print('\n' + '=' * 60)
    print('模型配置')
    print('=' * 60)
    print(f'  输入通道: {1 if use_grayscale else 3} ({"灰度图" if use_grayscale else "彩色图"})')
    print(f'  嵌入维度: {model.embed_dim}')
    print(f'  Encoder层数: {model.encoder.num_layers}')
    print(f'  注意力头数: {model.encoder.layers[0].attn_block.attn.num_heads}')
    print(f'  使用LS卷积: {model.encoder.layers[0].use_ls_conv}')
    print(f'  使用Kimi残差: {model.encoder.layers[0].attn_block.__class__.__name__}')
    print('=' * 60)

    print('\n创建数据加载器...')

    if os.path.exists(csv_file):
        print(f'找到 {csv_file}，使用真实数据集')

        train_loader, test_loader = create_fer2013_dataloaders(
            csv_file=csv_file,
            batch_size=32,
            num_workers=0,
            img_size=224
        )

        print(f'训练集: {len(train_loader.dataset)} 样本')
        print(f'测试集: {len(test_loader.dataset)} 样本')
        num_epochs = 50  # 真实数据集训练更多轮
    else:
        print(f'未找到 {csv_file}，使用合成数据集测试')
        print(f'提示: 将FER2013.csv放在项目根目录以使用真实数据')

        train_dataset = SyntheticFERDataset(num_samples=1000, num_classes=7)
        test_dataset = SyntheticFERDataset(num_samples=200, num_classes=7)

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        print(f'训练集: {len(train_dataset)} 样本')
        print(f'测试集: {len(test_dataset)} 样本')
        print(f'表情类别: {EMOTION_LABELS}')
        num_epochs = 5  # 合成数据集只训练5轮

    # 消融实验模式（可以修改进行对比实验）
    ablation_mode = {
        'use_ls_conv': True,  # LS卷积开关
        'use_kimi_residual': True  # Kimi残差开关
    }

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=3e-4,
        weight_decay=0.05,
        device=device,
        save_dir='./checkpoints',
        log_dir='./logs',
        use_mixed_precision=True,
        save_attention=True,
        save_confusion_matrix=True,
        ablation_mode=ablation_mode,
        log_batch_every=10  # 每隔10个batch记录详细信息
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()