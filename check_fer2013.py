"""
检查 FER2013.csv 文件
"""

import os
import pandas as pd

def check_fer2013(csv_file='FER2013.csv'):
    """检查FER2013数据集"""

    print("="*60)
    print("FER2013 数据集检查")
    print("="*60)

    if not os.path.exists(csv_file):
        print(f"\n✗ 未找到 {csv_file}")
        print(f"\n请将 FER2013.csv 文件放在项目根目录")
        return False

    print(f"\n✓ 找到 {csv_file}")
    print(f"  文件大小: {os.path.getsize(csv_file) / 1024 / 1024:.2f} MB")

    try:
        # 读取CSV
        df = pd.read_csv(csv_file)
        print(f"\n✓ 成功读取 CSV 文件")
        print(f"  数据形状: {df.shape}")

        # 检查列名
        print(f"\n列名: {list(df.columns)}")

        # 统计数据分布
        print(f"\n数据分布:")
        print(f"  总样本数: {len(df)}")

        if 'Usage' in df.columns:
            usage_counts = df['Usage'].value_counts()
            print(f"\n按Usage划分:")
            for usage, count in usage_counts.items():
                print(f"  {usage}: {count} 样本")

        if 'emotion' in df.columns:
            emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_counts = df['emotion'].value_counts().sort_index()
            print(f"\n按表情类别划分:")
            for emotion, count in emotion_counts.items():
                print(f"  {emotion} ({emotion_names[emotion]}): {count} 样本")

        # 检查第一行数据
        print(f"\n第一行数据示例:")
        first_row = df.iloc[0]
        print(f"  emotion: {first_row['emotion']}")
        if 'Usage' in first_row:
            print(f"  Usage: {first_row['Usage']}")
        if 'pixels' in first_row:
            pixels = first_row['pixels'].split()
            print(f"  pixels (前10个): {pixels[:10]}")
            print(f"  像素总数: {len(pixels)}")

        print(f"\n✓ 数据集检查通过")
        return True

    except Exception as e:
        print(f"\n✗ 读取CSV文件失败: {e}")
        return False

if __name__ == "__main__":
    check_fer2013()
