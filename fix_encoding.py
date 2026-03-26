"""
修复编码问题的脚本
"""

import os
import sys

# 修复编码
if sys.platform == 'win32':
    import locale
    # 设置控制台编码为 UTF-8
    try:
        locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    except:
        # Windows 可能没有中文 locale
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

print('✓ 编码已设置为 UTF-8')
print('现在可以运行训练了！')
print('\n运行命令：python train.py')

# 自动运行训练
import subprocess
result = subprocess.run([sys.executable, 'train.py'], shell=True)
