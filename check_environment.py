"""
环境检查与修复脚本
检查NumPy、PyTorch、CUDA等依赖的兼容性
"""

import sys
import subprocess
import os

def run_command(cmd):
    """运行命令并返回输出"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("Python版本检查")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print()

def check_pip_packages():
    """检查已安装的包"""
    print("=" * 60)
    print("已安装包版本")
    print("=" * 60)
    
    packages = ['numpy', 'torch', 'torchvision', 'opencv-python', 'pillow', 'tensorboard']
    
    for package in packages:
        stdout, stderr, code = run_command(f"pip show {package}")
        if code == 0:
            for line in stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    print(f"✓ {package}: {version}")
                    break
        else:
            print(f"✗ {package}: 未安装")
    print()

def check_numpy_compatibility():
    """检查NumPy兼容性"""
    print("=" * 60)
    print("NumPy兼容性检查")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
        print(f"  NumPy路径: {np.__file__}")
        
        # 测试NumPy是否正常工作
        test_array = np.array([1, 2, 3])
        print(f"  NumPy测试: ✓ 正常")
        
    except ImportError as e:
        print(f"✗ NumPy导入失败: {e}")
    except Exception as e:
        print(f"✗ NumPy错误: {e}")
    print()

def check_pytorch():
    """检查PyTorch"""
    print("=" * 60)
    print("PyTorch检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"  PyTorch路径: {torch.__file__}")
        
        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print("  ⚠ CUDA不可用，将使用CPU")
        
        # 测试PyTorch是否正常工作
        test_tensor = torch.randn(2, 3)
        print(f"  PyTorch测试: ✓ 正常")
        
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
    except Exception as e:
        print(f"✗ PyTorch错误: {e}")
    print()

def check_directory_issues():
    """检查目录问题"""
    print("=" * 60)
    print("目录问题检查")
    print("=" * 60)
    
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查路径中的特殊字符
    issues = []
    if ' ' in current_dir:
        issues.append("路径包含空格")
    if '(' in current_dir or ')' in current_dir:
        issues.append("路径包含括号")
    
    if issues:
        print("⚠ 发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n建议:")
        print("  1. 重命名目录，避免特殊字符")
        print("  2. 使用短路径或符号链接")
        print("  3. 在其他目录下运行代码")
    else:
        print("✓ 路径检查通过")
    print()

def check_duplicate_packages():
    """检查重复包"""
    print("=" * 60)
    print("重复包检查")
    print("=" * 60)
    
    stdout, stderr, code = run_command("pip list | grep -i numpy")
    
    if stdout:
        numpy_packages = [line.strip() for line in stdout.split('\n') if 'numpy' in line.lower()]
        if len(numpy_packages) > 1:
            print(f"⚠ 发现多个NumPy相关包:")
            for pkg in numpy_packages:
                print(f"  - {pkg}")
            print("\n建议:")
            print("  pip uninstall numpy numpy-base")
            print("  pip install numpy")
        else:
            print("✓ 无重复包")
    print()

def suggest_fix():
    """建议修复方案"""
    print("=" * 60)
    print("修复建议")
    print("=" * 60)
    
    print("\n方案1: 重新安装NumPy和PyTorch")
    print("  pip uninstall numpy torch torchvision -y")
    print("  pip install numpy torch torchvision")
    
    print("\n方案2: 指定兼容版本")
    print("  pip install numpy==1.24.3 torch==2.0.1 torchvision==0.15.2")
    
    print("\n方案3: 创建新环境")
    print("  conda create -n vitlsnet python=3.10")
    print("  conda activate vitlsnet")
    print("  pip install -r requirements.txt")
    
    print("\n方案4: 修复路径问题")
    print("  将项目移动到简单路径，如: C:/vit_lsnet/")
    print("  或在项目根目录运行: cd /d \"C:/vit_lsnet/\"")

def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "环境检查与修复工具" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # 运行各项检查
    check_python_version()
    check_pip_packages()
    check_numpy_compatibility()
    check_pytorch()
    check_directory_issues()
    check_duplicate_packages()
    
    # 给出建议
    suggest_fix()
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
