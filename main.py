"""
主程序入口 - 完整的ViT-LSNet人脸表情识别系统
"""

import torch
import numpy as np
from PIL import Image
import cv2

from vit_lsnet_fer import ViTLSNetFER, ViTLSNetFERConfig, EMOTION_LABELS
from mtcnn_detector import MTCNNDetector
from train import Trainer
from dataset import create_dataloaders
from torch.utils.data import DataLoader


class ExpressionRecognitionSystem:
    """表情识别系统"""
    
    def __init__(self, 
                 config_name: str = 'LIGHT',
                 checkpoint_path: str = None,
                 device: str = 'cuda'):
        """
        Args:
            config_name: 模型配置 ('TINY', 'LIGHT', 'BASE')
            checkpoint_path: 模型检查点路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.config_name = config_name
        
        # 创建模型
        config = getattr(ViTLSNetFERConfig, config_name)
        self.model = ViTLSNetFER(**config, use_mtcnn=True)
        self.model.to(device)
        self.model.eval()
        
        # MTCNN检测器
        self.mtcnn = MTCNNDetector()
        
        # 加载检查点
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            print(f'加载模型: {checkpoint_path}')
        else:
            print('使用随机初始化模型')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_acc' in checkpoint:
            print(f'最佳准确率: {checkpoint["best_acc"]:.2f}%')
    
    def predict_image(self, image, return_prob=False):
        """
        预测单张图像的表情
        
        Args:
            image: 图像 (numpy数组或PIL Image)
            return_prob: 是否返回概率
            
        Returns:
            pred_class: 预测类别
            pred_prob: 预测概率（可选）
        """
        # 预处理
        if isinstance(image, np.ndarray):
            # MTCNN预处理
            tensor = self.mtcnn.preprocess(image)
        elif isinstance(image, Image.Image):
            # PIL图像转numpy
            image_np = np.array(image)
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            tensor = self.mtcnn.preprocess(image_np)
        else:
            raise TypeError('不支持的图像类型')
        
        # 添加batch维度
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            pred_class, pred_prob = self.model.predict(tensor)
        
        pred_class = pred_class.cpu().item()
        pred_prob = pred_prob.cpu().squeeze().numpy()
        
        if return_prob:
            return pred_class, pred_prob
        else:
            return pred_class
    
    def predict_batch(self, images):
        """
        批量预测
        
        Args:
            images: 图像列表
            
        Returns:
            pred_classes: 预测类别列表
            pred_probs: 预测概率列表
        """
        pred_classes = []
        pred_probs = []
        
        for image in images:
            pred_class, pred_prob = self.predict_image(image, return_prob=True)
            pred_classes.append(pred_class)
            pred_probs.append(pred_prob)
        
        return np.array(pred_classes), np.array(pred_probs)
    
    def get_emotion_label(self, class_id: int) -> str:
        """获取表情标签"""
        return EMOTION_LABELS.get(class_id, f'Unknown({class_id})')
    
    def visualize_attention(self, image, layer_idx=-1, save_path=None):
        """
        可视化注意力图
        
        Args:
            image: 输入图像
            layer_idx: 层索引
            save_path: 保存路径（可选）
            
        Returns:
            attention_map: 注意力图
        """
        # 预处理
        if isinstance(image, np.ndarray):
            tensor = self.mtcnn.preprocess(image)
        else:
            image = np.array(image)
            tensor = self.mtcnn.preprocess(image)
        
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # 获取注意力图
        with torch.no_grad():
            attention_map = self.model.get_attention_maps(tensor, layer_idx)
        
        # 平均多头注意力
        attention_map = attention_map.mean(dim=1).squeeze(0)  # (N, N)
        
        # 使用CLS token的注意力
        cls_attention = attention_map[0, 1:]  # 移除CLS token
        
        # Reshape为图像尺寸
        H = W = int(np.sqrt(cls_attention.shape[0]))
        attention_img = cls_attention.reshape(H, W).cpu().numpy()
        
        # 归一化到[0, 255]
        attention_img = (attention_img - attention_img.min()) / \
                       (attention_img.max() - attention_img.min() + 1e-8) * 255
        attention_img = attention_img.astype(np.uint8)
        
        # 应用彩色映射
        attention_img = cv2.applyColorMap(attention_img, cv2.COLORMAP_JET)
        
        # Resize到原图尺寸
        if isinstance(image, np.ndarray):
            H_orig, W_orig = image.shape[:2]
            attention_img = cv2.resize(attention_img, (W_orig, H_orig))
        
        # 叠加到原图
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(image, 0.6, attention_img, 0.4, 0)
        else:
            overlay = attention_img
        
        if save_path:
            cv2.imwrite(save_path, overlay)
        
        return overlay


def demo():
    """演示程序"""
    print("=" * 60)
    print("ViT-LSNet 人脸表情识别系统")
    print("=" * 60)
    
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建系统
    print("\n创建模型...")
    system = ExpressionRecognitionSystem(
        config_name='LIGHT',
        device=device
    )
    
    # 创建测试图像
    print("\n创建测试图像...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 预测
    print("预测表情...")
    pred_class, pred_prob = system.predict_image(test_image, return_prob=True)
    pred_label = system.get_emotion_label(pred_class)
    
    print(f"\n预测结果:")
    print(f"表情类别: {pred_class} ({pred_label})")
    print(f"置信度: {pred_prob[pred_class]:.4f}")
    print(f"\n所有表情概率:")
    for class_id, prob in enumerate(pred_prob):
        label = system.get_emotion_label(class_id)
        print(f"  {label}: {prob:.4f}")
    
    # 可视化注意力
    print("\n生成注意力图...")
    attention_map = system.visualize_attention(test_image)
    print(f"注意力图形状: {attention_map.shape}")
    
    print("\n演示完成!")
    print("\n提示:")
    print("- 使用真实人脸图像进行测试")
    print("- 训练模型以获得准确预测")
    print("- 调整配置以适应不同硬件")


def train_model():
    """训练模型"""
    print("=" * 60)
    print("训练ViT-LSNet模型")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    from dataset import SyntheticFERDataset
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = ViTLSNetFER(**ViTLSNetFERConfig.LIGHT)
    
    # 创建数据加载器
    train_dataset = SyntheticFERDataset(num_samples=500, num_classes=7)
    test_dataset = SyntheticFERDataset(num_samples=100, num_classes=7)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=3,
        device=device,
        save_dir='./checkpoints',
        log_dir='./logs'
    )
    
    # 开始训练
    trainer.train()
    
    print("\n训练完成!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # 训练模式
        train_model()
    else:
        # 演示模式
        demo()
