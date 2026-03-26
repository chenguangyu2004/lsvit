"""
数据加载器 - 支持FER-2013、RAF-DB、CK+等表情数据集
支持灰度图和彩色图
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import json
from typing import Optional, Tuple, List, Dict
import cv2


class FER2013Dataset(Dataset):
    """
    FER-2013 数据集加载器

    直接从 FER2013.csv 文件加载数据
    数据格式: emotion, pixels, Usage
    """

    def __init__(self,
                 csv_file: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 img_size: int = 224):
        """
        Args:
            csv_file: FER2013.csv 文件路径
            split: 'train', 'test', 或 'all'
            transform: 图像变换
            img_size: 目标图像尺寸（从48x48 resize到img_size）
        """
        self.csv_file = csv_file
        self.split = split
        self.img_size = img_size

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

        # 加载数据集
        self.data = self._load_data()
        print(f"  加载 FER2013 {split} 数据集: {len(self.data)} 样本")

    def _load_data(self) -> List[Tuple[Image.Image, int]]:
        """从CSV文件加载数据"""
        import pandas as pd

        # 读取CSV文件
        df = pd.read_csv(self.csv_file)

        # 根据 split 筛选数据
        if self.split == 'train':
            df = df[df['Usage'] == 'Training']
        elif self.split == 'test':
            df = df[(df['Usage'] == 'PublicTest') | (df['Usage'] == 'PrivateTest')]
        # 'all' 则使用全部数据

        data = []

        for idx, row in df.iterrows():
            # 解析像素值
            pixels_str = row['pixels']
            emotion = int(row['emotion'])

            # 将字符串转为数组
            pixels = np.array([int(p) for p in pixels_str.split()], dtype=np.uint8)

            # reshape 为 48x48
            image = pixels.reshape(48, 48)

            # 转为 PIL Image（灰度图）
            pil_image = Image.fromarray(image, mode='L')

            data.append((pil_image, emotion))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


def create_fer2013_dataloaders(csv_file: str,
                               batch_size: int = 32,
                               num_workers: int = 0,
                               img_size: int = 224) -> Tuple[DataLoader, DataLoader]:
    """
    创建 FER2013 数据加载器（从CSV文件）

    Args:
        csv_file: FER2013.csv 文件路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸

    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 测试集变换
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 训练集
    train_dataset = FER2013Dataset(
        csv_file=csv_file,
        split='train',
        transform=train_transform,
        img_size=img_size
    )

    # 测试集
    test_dataset = FER2013Dataset(
        csv_file=csv_file,
        split='test',
        transform=test_transform,
        img_size=img_size
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader



class FERDataset(Dataset):
    """
    通用表情识别数据集
    
    支持的数据集:
    - FER-2013: 灰度图, 7类表情
    - RAF-DB: 彩色图, 7类表情
    - CK+: 灰度图/彩色图, 7类表情
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 is_grayscale: bool = False,
                 img_size: int = 224):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train' 或 'test'
            transform: 图像变换
            is_grayscale: 是否为灰度图
            img_size: 目标图像尺寸
        """
        self.data_dir = data_dir
        self.split = split
        self.is_grayscale = is_grayscale
        self.img_size = img_size
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5] if not is_grayscale else [0.5],
                    std=[0.5, 0.5, 0.5] if not is_grayscale else [0.5]
                )
            ])
        else:
            self.transform = transform
        
        # 加载数据集
        self.samples = self._load_dataset()
        
        print(f"加载 {split} 数据集: {len(self.samples)} 样本")
    
    def _load_dataset(self) -> List[Tuple[str, int]]:
        """加载数据集样本列表"""
        samples = []
        
        # 根据数据集格式加载
        if 'fer2013' in self.data_dir.lower():
            samples = self._load_fer2013()
        elif 'raf-db' in self.data_dir.lower():
            samples = self._load_rafdb()
        elif 'ck+' in self.data_dir.lower():
            samples = self._load_ckplus()
        else:
            # 通用格式: 按类别组织的文件夹结构
            samples = self._load_generic()
        
        return samples
    
    def _load_fer2013(self) -> List[Tuple[str, int]]:
        """加载FER-2013数据集"""
        samples = []
        
        # FER-2013通常是CSV格式或HDF5格式
        csv_path = os.path.join(self.data_dir, f'fer2013_{self.split}.csv')
        
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                image = pixels.reshape(48, 48)
                label = row['emotion']
                
                # 保存为临时图像文件或直接存储像素
                temp_path = f'temp_{self.split}_{idx}.png'
                cv2.imwrite(temp_path, image)
                samples.append((temp_path, label))
        
        return samples
    
    def _load_rafdb(self) -> List[Tuple[str, int]]:
        """加载RAF-DB数据集"""
        samples = []
        
        # RAF-DB通常是按类别组织的文件夹结构
        label_file = os.path.join(self.data_dir, f'{self.split}_labels.txt')
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_name = parts[0]
                    label = int(parts[1])
                    img_path = os.path.join(self.data_dir, 'images', self.split, img_name)
                    if os.path.exists(img_path):
                        samples.append((img_path, label))
        
        return samples
    
    def _load_ckplus(self) -> List[Tuple[str, int]]:
        """加载CK+数据集"""
        samples = []
        
        # CK+通常是按表情序列组织
        split_dir = os.path.join(self.data_dir, self.split)
        
        if os.path.exists(split_dir):
            # 假设每个表情序列有一个标签文件
            for emotion_dir in os.listdir(split_dir):
                emotion_path = os.path.join(split_dir, emotion_dir)
                if os.path.isdir(emotion_path):
                    label = int(emotion_dir)
                    
                    # 使用序列中的最后一帧
                    image_files = sorted([f for f in os.listdir(emotion_path) 
                                        if f.endswith(('.png', '.jpg', '.jpeg'))])
                    if image_files:
                        img_path = os.path.join(emotion_path, image_files[-1])
                        samples.append((img_path, label))
        
        return samples
    
    def _load_generic(self) -> List[Tuple[str, int]]:
        """加载通用格式数据集（按类别文件夹组织）"""
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        if os.path.exists(split_dir):
            for label_dir in sorted(os.listdir(split_dir)):
                label_path = os.path.join(split_dir, label_dir)
                if os.path.isdir(label_path):
                    label = int(label_dir)
                    for img_file in os.listdir(label_path):
                        if img_file.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(label_path, img_file)
                            samples.append((img_path, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图像
        if self.is_grayscale:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_augmentation(is_grayscale: bool = False, 
                         img_size: int = 224,
                         is_train: bool = True):
    """
    获取数据增强
    
    Args:
        is_grayscale: 是否为灰度图
        img_size: 图像尺寸
        is_train: 是否为训练集
        
    Returns:
        transform: 变换组合
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5] if not is_grayscale else [0.5],
                std=[0.5, 0.5, 0.5] if not is_grayscale else [0.5]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5] if not is_grayscale else [0.5],
                std=[0.5, 0.5, 0.5] if not is_grayscale else [0.5]
            )
        ])


def create_dataloaders(data_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       is_grayscale: bool = False,
                       img_size: int = 224) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        is_grayscale: 是否为灰度图
        img_size: 图像尺寸
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 训练集
    train_dataset = FERDataset(
        data_dir=data_dir,
        split='train',
        transform=get_data_augmentation(is_grayscale, img_size, is_train=True),
        is_grayscale=is_grayscale,
        img_size=img_size
    )
    
    # 测试集
    test_dataset = FERDataset(
        data_dir=data_dir,
        split='test',
        transform=get_data_augmentation(is_grayscale, img_size, is_train=False),
        is_grayscale=is_grayscale,
        img_size=img_size
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


class SyntheticFERDataset(Dataset):
    """
    合成表情数据集（用于测试代码）
    """
    
    def __init__(self,
                 num_samples: int = 1000,
                 num_classes: int = 7,
                 img_size: int = 224,
                 is_grayscale: bool = False):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.is_grayscale = is_grayscale
        
        # 预生成随机标签
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机图像
        channels = 1 if self.is_grayscale else 3
        image = torch.randn(channels, self.img_size, self.img_size)
        label = self.labels[idx]
        
        return image, label


# 测试代码
if __name__ == "__main__":
    print("测试数据加载器...")
    
    # 创建合成数据集用于测试
    train_dataset = SyntheticFERDataset(num_samples=100, num_classes=7)
    test_dataset = SyntheticFERDataset(num_samples=50, num_classes=7)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试数据加载
    print("\n测试数据加载:")
    for images, labels in train_loader:
        print(f"批次形状: {images.shape}, 标签形状: {labels.shape}")
        print(f"标签范围: {labels.min().item()} - {labels.max().item()}")
        break
    
    # 测试灰度图
    print("\n测试灰度图数据集:")
    gray_dataset = SyntheticFERDataset(num_samples=50, is_grayscale=True)
    gray_loader = DataLoader(gray_dataset, batch_size=8, shuffle=True)
    
    for images, labels in gray_loader:
        print(f"灰度图批次形状: {images.shape}")
        break
    
    print("\n数据加载器测试通过!")
