"""
修复NumPy版本兼容性问题
将NumPy降级到1.x以兼容PyTorch
"""

import subprocess
import sys

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"执行: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode == 0:
        print("✓ 执行成功")
    else:
        print(f"✗ 执行失败，返回码: {result.returncode}")
    
    return result.returncode == 0

def main():
    """主函数"""
    print("="*60)
    print("  NumPy版本兼容性修复工具")
    print("="*60)
    
    print("\n当前问题:")
    print("  - NumPy 2.0.2 已安装")
    print("  - PyTorch 是用 NumPy 1.x 编译的")
    print("  - 版本不兼容导致错误")
    
    print("\n" + "="*60)
    print("修复方案")
    print("="*60)
    
    # 方案1: 降级NumPy
    print("\n方案1: 降级NumPy到1.x（推荐）")
    print("\n步骤:")
    
    print("\n1. 卸载当前NumPy")
    run_command(
        "pip uninstall numpy -y",
        "卸载NumPy 2.0.2"
    )
    
    print("\n2. 安装NumPy 1.26.4（最后一个1.x稳定版本）")
    run_command(
        "pip install numpy==1.26.4",
        "安装NumPy 1.26.4"
    )
    
    print("\n3. 验证NumPy版本")
    run_command(
        "python -c \"import numpy as np; print('NumPy版本:', np.__version__)\"",
        "验证NumPy版本"
    )
    
    # 方案2: 升级PyTorch
    print("\n" + "="*60)
    print("方案2: 升级到支持NumPy 2.0的PyTorch（可选）")
    print("="*60)
    
    print("\n如果方案1无效，可以尝试升级PyTorch:")
    print("  pip install --upgrade torch torchvision")
    print("\n或使用支持NumPy 2.0的PyTorch版本:")
    print("  pip install torch>=2.4.0")
    
    # 方案3: 重新创建环境
    print("\n" + "="*60)
    print("方案3: 创建新的conda环境（最干净）")
    print("="*60)
    
    print("\n如果上述方案都无效，建议创建新环境:")
    print("\n1. 创建新环境:")
    print("   conda create -n vitlsnet python=3.10 -y")
    print("\n2. 激活环境:")
    print("   conda activate vitlsnet")
    print("\n3. 安装PyTorch（CUDA 11.8）:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n4. 安装NumPy（会自动安装兼容版本）:")
    print("   pip install numpy")
    print("\n5. 安装其他依赖:")
    print("   pip install -r requirements.txt")
    
    # 验证修复
    print("\n" + "="*60)
    print("验证修复")
    print("="*60)
    
    print("\n现在测试NumPy和PyTorch是否能正常工作:")
    
    success = run_command(
        "python -c \"import numpy as np; import torch; print('✓ NumPy:', np.__version__); print('✓ PyTorch:', torch.__version__); print('✓ GPU:', torch.cuda.is_available())\"",
        "测试NumPy和PyTorch"
    )
    
    if success:
        print("\n" + "="*60)
        print("✓ 修复成功！")
        print("="*60)
        print("\n现在可以运行训练脚本了:")
        print("  python train.py")
    else:
        print("\n" + "="*60)
        print("✗ 修复失败")
        print("="*60)
        print("\n请尝试方案2或方案3")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
