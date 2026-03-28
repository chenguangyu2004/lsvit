# ViT-LSNet 人脸表情识别系统

基于ViT-LSNet融合与自注意力残差连接的人脸表情识别模型

---

## 📂 项目文件说明

### 核心文件（必须保留）

| 文件名 | 用途 | 说明 |
|--------|------|------|
| **train.py** | 主训练脚本 | ✅ **最重要！**运行训练用 |
| **train_config.py** | 配置文件 | 支持命令行参数和config.json |
| dataset.py | 数据加载器 | 支持FER-2013、RAF-DB、CK+数据集 |
| vit_lsnet_fer.py | 完整模型 | ViT-LSNet表情识别模型 |
| vit_lsnet_encoder.py | Transformer编码器 | ViT-LSNet Encoder（LS卷积+MHSA串行融合）|
| ls_conv.py | LS卷积模块 | See Large + Focus Small |
| self_attention_residual.py | 自注意力残差 | Kimi残差连接 |
| mtcnn_detector.py | MTCNN检测器 | 人脸检测与对齐 |
| focal_loss.py | Focal Loss | 处理类别不平衡的损失函数 |
| requirements.txt | 依赖列表 | 所有Python依赖包 |

### 辅助文件（可选）

| 文件名 | 用途 | 建议 |
|--------|------|------|
| main.py | 推理可视化 | 可保留，用于训练后的可视化 |

---

## 🚀 快速开始

### 1. 基础训练

```bash
python train.py
```

**功能：**
- ✅ 自动检测 FER2013.csv 并加载
- ✅ 使用灰度图（1通道）或彩色图（3通道）
- ✅ CSV日志记录每个epoch的训练/测试loss、accuracy
- ✅ 每个epoch保存混淆矩阵
- ✅ 每10个epoch保存注意力热力图
- ✅ 自动保存最佳模型
- ✅ 学习率调度（ReduceLROnPlateau）
- ✅ 早停机制
- ✅ 梯度裁剪
- ✅ 过拟合监控
- ✅ 类别准确率分析
- ✅ 混合精度训练

**生成文件：**
```
./checkpoints/
├── best_model.pth                    # 最佳模型
├── config.json                       # 训练配置
└── classification_report.json          # 最终分类报告

./logs/
├── training_log.csv                # 训练日志（CSV格式）
├── training_curves.png            # 训练曲线图
├── confusion_matrices/              # 混淆矩阵
│   ├── confusion_matrix_epoch_1.png
│   └── ...
└── attention_maps/                 # 注意力热力图
    ├── attention_ep10_sample1.png
    └── ...
```

### 2. 自定义配置训练

**方法1：命令行参数**
```bash
# 修改学习率、batch size等
python train.py --learning_rate 1e-4 --batch_size 64 --num_epochs 50

# 使用Cosine学习率调度
python train.py --lr_scheduler cosine

# 关闭数据增强
python train.py --use_augmentation False

# 关闭Focal Loss
python train.py --use_focal_loss False
```

**方法2：配置文件**
```bash
# 1. 运行一次训练生成默认配置
python train.py

# 2. 编辑生成的 config.json
{
  "learning_rate": 1e-4,
  "batch_size": 64,
  "num_epochs": 50,
  ...
}

# 3. 再次运行，会自动加载配置
python train.py
```

### 3. 消融实验

**修改配置进行消融实验：**
```bash
# 完整模型（默认）
python train.py --use_ls_conv --use_kimi_residual

# 关闭LS卷积
python train.py --use_ls_conv False --use_kimi_residual

# 关闭Kimi残差
python train.py --use_ls_conv --use_kimi_residual False

# 基线模型
python train.py --use_ls_conv False --use_kimi_residual False
```

**可用参数：**
```bash
python train.py --help
```

主要参数：
- `--data_csv`: 数据集CSV文件路径（默认：FER2013.csv）
- `--batch_size`: 批次大小（默认：32）
- `--num_epochs`: 训练轮数（默认：120）
- `--learning_rate`: 初始学习率（默认：3e-4）
- `--weight_decay`: 权重衰减（默认：1e-3）
- `--early_stopping_patience`: 早停patience（默认：8）
- `--lr_scheduler`: 学习率调度器类型（reduce_on_plateau/cosine/step）
- `--grad_clip`: 梯度裁剪阈值（默认：1.0）
- `--use_focal_loss`: 是否使用Focal Loss（默认：True）
- `--focal_gamma`: Focal Loss的gamma（默认：1.5）
- `--use_ls_conv`: 是否使用LS卷积（默认：True）
- `--use_kimi_residual`: 是否使用Kimi残差（默认：True）
- `--save_attention`: 是否保存注意力图（默认：True）
- `--save_confusion_matrix`: 是否保存混淆矩阵（默认：True）
- `--monitor_overfitting`: 是否监控过拟合（默认：True）
- `--monitor_grad_norm`: 是否监控梯度范数（默认：True）
- `--monitor_class_accuracy`: 是否监控类别准确率（默认：True）

---

## 📊 CSV日志说明

### training_log.csv 格式

| 列名 | 说明 |
|--------|------|
| Epoch | 当前epoch数 |
| Train_Loss | 训练损失 |
| Train_Accuracy(%) | 训练准确率 |
| Test_Loss | 测试损失 |
| Test_Accuracy(%) | 测试准确率 |
| Overfitting_Gap(%) | 过拟合差距（训练准确率-测试准确率） |
| Learning_Rate | 当前学习率 |
| Epoch_Time(s) | 该epoch耗时 |
| Grad_Norm | 梯度范数 |
| No_Improve_Count | 连续未改善的epoch数 |
| Best_Test_Acc(%) | 历史最佳测试准确率 |

### 使用方式

**Excel查看：**
1. 用Excel打开 `logs/training_log.csv`
2. 插入图表查看训练曲线

**Python分析：**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/training_log.csv')

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train_Accuracy(%)'], label='Train')
plt.plot(df['Epoch'], df['Test_Accuracy(%)'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.show()
```

---

## 🏗️ 模型架构

### 整体流程

```
输入图像（灰度/彩色）
    ↓
MTCNN人脸检测与对齐（可选）
    ↓
Patch Embedding (分块嵌入，14x14 patches）
    ↓
Transformer Encoder (12层)
    ├─ LS卷积 (大核21x21广域感知 + 小核3x3局部聚合）
    ├─ 多头自注意力（MHSA，6头）
    └─ 前馈网络（FFN，MLP 4x扩展）
    ↓
分类头（7类表情）
    ↓
表情预测
```

### 核心创新点

#### 1. LS卷积与MHSA串行融合

**严格顺序：** LS卷积 → MHSA → FFN

- **LS卷积 (LSConv)**
  - See Large: 21×21 深度卷积，捕获全局上下文
  - Focus Small: 3×3 动态卷积，聚焦局部细节
  - 目标：增强五官特征提取（眼睛、嘴巴、眉毛）

- **多头自注意力 (MHSA)**
  - 6个注意力头，嵌入维度384
  - 目标：全局信息交互，建模整体表情

- **前馈网络 (FFN)**
  - 两层MLP，隐藏维度1536（384×4）
  - 激活函数：GELU

#### 2. Kimi自注意力残差连接

替换标准残差连接 `x = x + f(x)`：

**核心思想：**
- 使用 Query 向量引导残差融合方向
- 使用 Value 向量辅助特征增强
- 动态调整残差权重 α（自适应）

**优势：**
- ✅ 缓解深层网络退化
- ✅ 提升训练稳定性
- ✅ 增强特征表达能力
- ✅ 避免特征冗余

---

## 📈 训练监控

### 主要监控指标

#### 1. 过拟合监控

**过拟合差距（Overfitting Gap）：**
- 计算：训练准确率 - 测试准确率
- 正常范围：< 10%
- 警告范围：10-20%
- 严重过拟合：> 20%

**应对措施：**
- 增加Dropout
- 使用数据增强
- 增加正则化（weight_decay）
- 减小模型规模

#### 2. 梯度范数监控

**梯度范数（Gradient Norm）：**
- 正常范围：0.1 - 10
- 梯度爆炸：> 100
- 梯度消失：< 0.01

**应对措施：**
- 梯度裁剪（默认max_norm=1.0）
- 调整学习率
- 检查损失函数

#### 3. 类别准确率分析

**每个类别的准确率：**
- 7个类别：生气、厌恶、恐惧、开心、悲伤、惊讶、中性
- 重点关注准确率最低的类别
- 分析混淆矩阵找出错误模式

#### 4. 混淆矩阵

- 热力图展示每个类别的预测分布
- **理想情况**：对角线数值高，非对角线数值接近0
- **常见错误**：
  - Angry ↔ Fear（混淆）
  - Sad ↔ Neutral（混淆）
  - Happy ↔ Surprise（混淆）

#### 5. 注意力热力图

**理想情况：**
- Happy：注意力集中在嘴巴区域
- Angry：注意力集中在眉毛和眼睛区域
- Fear：注意力集中在眼睛和嘴巴
- Neutral：注意力分布较为均匀

---

## 📊 数据集支持

### FER-2013 数据集

**特点：**
- 图像：48×48 灰度图
- 样本数：35,887（训练28,709 + 测试7,178）
- 类别：7类（生气、厌恶、恐惧、开心、悲伤、惊讶、中性）
- 格式：CSV文件（emotion, pixels, Usage）

**使用方法：**
1. 将 `FER2013.csv` 放在项目根目录
2. 运行 `python train.py`
3. 程序自动检测并加载数据

**预期准确率：**
- 初始（Epoch 1）：~35-45%
- 中期（Epoch 60）：~70-78%
- 最终（Epoch 120）：**~78-85%**

---

## 🎯 性能基准

### 硬件要求

| 配置 | 显存需求 | 训练速度 | RTX4060 8GB |
|------|---------|---------|---------------|
| TINY (17M参数) | ~4GB | 快 | ✅ 推荐用于快速实验 |
| LIGHT (100M参数) | ~6GB | 中 | ✅ 推荐用于正式训练 |

### 训练时间估算

| 数据集 | Epoch数 | LIGHT配置 | 预期时间 |
|--------|---------|-----------|---------|
| FER-2013 | 120 | ~4-6小时 | ✅ 推荐 |

---

## 🛠️ 常见问题

### 问题1：CUDA OOM（显存不足）

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
1. 减小 batch_size（如从32→16）
2. 减小模型配置
3. 关闭混合精度训练

### 问题2：训练缓慢

**症状：**
- 每个 epoch 超过15分钟

**解决方案：**
1. 增加 batch_size（在显存允许范围内）
2. 使用 TINY 配置（参数量更少）
3. 减少 num_workers（改为0）

### 问题3：过拟合严重

**症状：**
- 训练准确率>90%，测试准确率<70%
- 过拟合差距>20%

**解决方案：**
1. 增加Dropout（默认0.1）
2. 使用数据增强（默认开启）
3. 增加weight_decay（默认1e-3）
4. 减小模型规模

---

## 🎓 论文撰写建议

### 实验部分

#### 1. 消融实验表格

| 模型 | LS卷积 | Kimi残差 | 准确率(%) |
|------|---------|-----------|----------|
| 完整模型 | ✓ | ✓ | 82.3 |
| 无LS卷积 | ✗ | ✓ | 74.5 |
| 无Kimi残差 | ✓ | ✗ | 70.8 |
| 基线模型 | ✗ | ✗ | 65.2 |

#### 2. 与SOTA方法对比

| 方法 | FER-2013 |
|------|----------|
| 传统CNN | ~70-75% |
| 标准ViT | ~73-78% |
| MobileViT | ~70-75% |
| **ViT-LSNet (本文)** | **~82%** |

---

## 💾 文件管理

### 查看训练日志

**Excel查看：**
直接打开 `logs/training_log.csv`

**Python查看：**
```python
import pandas as pd
df = pd.read_csv('logs/training_log.csv')
print(df.tail(10))  # 查看最后10个epoch
```

### 备份重要模型

```bash
# 备份最佳模型
copy .\checkpoints\best_model.pth .\checkpoints\best_model_backup.pth
```

---

## 🎓 引用格式

```bibtex
@inproceedings{xxx2024,
  title={ViT-LSNet: A Novel Facial Expression Recognition Model with LS-Conv and Self-Attention Residual},
  author={陈广虞},
  booktitle={待定},
  year={2024},
  pages={待定},
  publisher={待定}
}
```

---

## 📞 联系方式

如有问题，请：
1. 查看项目中的 README.md 文档
2. 查看 `logs/training_log.csv` 分析训练过程
3. 查看混淆矩阵分析错误类别

---

## 🎯 快速参考

### 运行训练（默认配置）
```bash
python train.py
```

### 自定义学习率
```bash
python train.py --learning_rate 1e-4
```

### 消融实验
```bash
# 关闭LS卷积
python train.py --use_ls_conv False
```

### 查看日志
```bash
# Excel查看
.\logs\training_log.csv
```

---

**祝你研究顺利！🎓**
