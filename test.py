import torch
import cv2
import numpy as np

print("Python解释器路径:", __file__)
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("OpenCV版本:", cv2.__version__)
print("NumPy版本:", np.__version__)  # 应显示1.x.x