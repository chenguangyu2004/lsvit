"""
MTCNN人脸检测与预处理模块
支持灰度图和彩色图输入
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch

class MTCNNDetector:
    """MTCNN人脸检测器"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 margin: int = 20,
                 min_face_size: int = 20):
        """
        Args:
            target_size: 目标图像尺寸 (H, W)
            margin: 人脸边界框扩展像素
            min_face_size: 最小人脸尺寸
        """
        self.target_size = target_size
        self.margin = margin
        self.min_face_size = min_face_size
        
        # 这里使用OpenCV的Haar级联作为MTCNN的轻量替代
        # 实际使用时可以替换为真正的MTCNN实现
        # from mtcnn import MTCNN
        # self.detector = MTCNN()
        
        # 使用OpenCV Haar Cascade作为演示
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测人脸位置
        
        Args:
            image: 输入图像 (BGR或灰度)
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        # 转为灰度图用于检测
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        # faces 可能是 numpy array 或者 tuple，确保转换为 list
        if isinstance(faces, np.ndarray):
            return faces.tolist()
        elif faces is None or len(faces) == 0:
            return []
        else:
            return list(faces)
    
    def align_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        对齐人脸（简化版，仅裁剪）
        
        Args:
            image: 输入图像
            bbox: 人脸边界框 (x, y, w, h)
            
        Returns:
            对齐后的人脸图像
        """
        x, y, w, h = bbox
        
        # 添加边界扩展
        x = max(0, x - self.margin)
        y = max(0, y - self.margin)
        w = min(image.shape[1] - x, w + 2 * self.margin)
        h = min(image.shape[0] - y, h + 2 * self.margin)
        
        # 裁剪人脸
        face = image[y:y+h, x:x+w]
        
        return face
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        完整预处理流程: 检测 -> 对齐 -> Resize -> 归一化
        
        Args:
            image: 输入图像 (BGR或灰度)
            
        Returns:
            预处理后的张量 (C, H, W)
        """
        # 检测人脸
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            # 如果未检测到人脸,使用整张图像
            face_img = image
        else:
            # 使用最大的人脸
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            face_img = self.align_face(image, faces[0])
        
        # Resize到目标尺寸
        face_img = cv2.resize(face_img, (self.target_size[1], self.target_size[0]))
        
        # 转换为RGB（如果是灰度图，复制3通道）
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        else:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # 转换为张量并归一化到[0,1]
        face_tensor = torch.from_numpy(face_img).float()
        face_tensor = face_tensor.permute(2, 0, 1) / 255.0
        
        return face_tensor
    
    def batch_preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        批量预处理
        
        Args:
            images: 图像列表
            
        Returns:
            批量张量 (B, C, H, W)
        """
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors, dim=0)


# 测试代码
if __name__ == "__main__":
    detector = MTCNNDetector()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试检测
    faces = detector.detect_faces(test_image)
    print(f"检测到 {len(faces)} 个人脸")
    
    # 测试预处理
    tensor = detector.preprocess(test_image)
    print(f"预处理后张量形状: {tensor.shape}")
    
    # 测试灰度图
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    tensor_gray = detector.preprocess(gray_image)
    print(f"灰度图预处理后张量形状: {tensor_gray.shape}")
