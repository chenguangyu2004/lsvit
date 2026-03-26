"""
环境检查与修复脚本
检查NumPy、PyTorch、CUDA等依赖的兼容性
"""

import sys
import os
import platform

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_python():
    """检查Python版本"""
    print_section("Python环境检查")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"平台: {platform.system()} {platform.release()}")
    print(f"处理器架构: {platform.machine()}")

def check_numpy():
    """检查NumPy"""
    print_section("NumPy检查")
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
        print(f"  NumPy路径: {os.path.dirname(np.__file__)}")
        
        # 测试NumPy功能
        test_arr = np.array([1, 2, 3])
        print(f"  测试数组: {test_arr}")
        
        return np.__version__
    except ImportError as e:
        print(f"✗ NumPy导入失败: {e}")
        return None
    except Exception as e:
        print(f"✗ NumPy运行时错误: {e}")
        return None

def check_torch():
    """检查PyTorch"""
    print_section("PyTorch检查")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"  PyTorch路径: {os.path.dirname(torch.__file__)}")
        
        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name}")
                print(f"      总显存: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("  ⚠ CUDA不可用，将使用CPU")
        
        return torch.__version__
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return None
    except Exception as e:
        print(f"✗ PyTorch运行时错误: {e}")
        return None

def check_torchvision():
    """检查torchvision"""
    print_section("TorchVision检查")
    try:
        import torchvision
        print(f"✓ TorchVision版本: {torchvision.__version__}")
        return torchvision.__version__
    except ImportError as e:
        print(f"✗ TorchVision导入失败: {e}")
        print("  安装: pip install torchvision")
        return None
    except Exception as e:
        print(f"✗ TorchVision运行时错误: {e}")
        return None

def check_opencv():
    """检查OpenCV"""
    print_section("OpenCV检查")
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        return cv2.__version__
    except ImportError as e:
        print(f"✗ OpenCV导入失败: {e}")
        print("  安装: pip install opencv-python")
        return None
    except Exception as e:
        print(f"✗ OpenCV运行时错误: {e}")
        return None

def check_project_path():
    """检查项目路径"""
    print_section("项目路径检查")
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查路径问题
    issues = []
    if ' ' in current_dir:
        issues.append("路径包含空格")
    if any(ord(c) > 127 for c in current_dir):
        issues.append("路径包含非ASCII字符（中文等）")
    
    if issues:
        print("⚠ 检测到路径问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n建议:")
        print("  1. 将项目移动到不含空格和中文的路径")
        print("     例如: C:/projects/vit_lsnet")
        print("  2. 或使用符号链接:")
        print("     mklink /D C:\\vit_lsnet \"{current_dir}\"")
    else:
        print("✓ 路径检查通过")

def check_gpu():
    """检查GPU"""
    print_section("GPU信息")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ GPU可用")
            
            # 显示GPU详细信息
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                
                print(f"  显存总量: {props.total_memory / 1024**3:.2f} GB")
                print(f"  计算能力: {props.major}.{props.minor}")
                print(f"  多处理器数量: {props.multi_processor_count}")
                
                # 显示当前显存使用
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                cached = torch.cuda.memory_reserved(i) / 1024**2
                print(f"  已分配显存: {allocated:.2f} MB")
                print(f"  已缓存显存: {cached:.2f} MB")
        else:
            print("✗ GPU不可用")
            print("  请检查:")
            print("  1. NVIDIA驱动是否正确安装")
            print("  2. CUDA和PyTorch版本是否匹配")
            print("  3. 是否安装了支持CUDA的PyTorch版本")
    except Exception as e:
        print(f"✗ GPU检查失败: {e}")

def fix_common_issues():
    """提供常见问题修复建议"""
    print_section("常见问题修复建议")
    
    print("1. NumPy导入失败:")
    print("   pip uninstall numpy")
    print("   pip install numpy")
    
    print("\n2. PyTorch导入失败:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n3. CUDA不可用:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n4. 路径包含特殊字符:")
    print("   将项目移动到简单路径，如:")
    print("   C:/projects/vit_lsnet")
    
    print("\n5. 清理缓存:")
    print("   pip cache purge")
    print("   python -m pip install --upgrade pip")

def main():
    """主函数"""
    print("=" * 60)
    print("  ViT-LSNet 环境检查工具")
    print("=" * 60)
    
    # 运行所有检查
    check_python()
    numpy_version = check_numpy()
    torch_version = check_torch()
    check_torchvision()
    check_opencv()
    check_project_path()
    check_gpu()
    fix_common_issues()
    
    # 总结
    print_section("总结")
    
    all_ok = True
    if numpy_version:
        print(f"✓ NumPy: {numpy_version}")
    else:
        print("✗ NumPy: 未安装或出错")
        all_ok = False
    
    if torch_version:
        print(f"✓ PyTorch: {torch_version}")
    else:
        print("✗ PyTorch: 未安装或出错")
        all_ok = False
    
    if all_ok:
        print("\n✓ 环境检查通过，可以开始训练！")
        print("\n运行训练:")
        print("  python train.py")
    else:
        print("\n✗ 环境存在问题，请根据上述建议修复")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
