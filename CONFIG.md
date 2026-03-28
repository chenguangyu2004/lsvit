# ViT-LSNet 训练配置说明

## 📋 快速配置指南

根据训练分析和硬件环境（4060显卡），提供常用配置方案。

---

## 🚀 基础配置（推荐用于4060显卡）

### 方案A: 保守训练（推荐）

```python
# main.py 配置
num_epochs = 100
learning_rate = 3e-4
weight_decay = 3e-4
batch_size = 32
early_stopping_patience = 8
use_focal_loss = True
class_weights = None  # 不使用类别权重
dropout_rate = 0.1

# 模型配置
num_layers = 12
embed_dim = 384
```

**预期效果**:
- 测试准确率: 67-70%
- 训练时间: ~5-6小时
- 过拟合差距: 10-15%
- 显存占用: ~4-5GB

---

### 方案B: 激进训练（实验）

```python
# main.py 配置
num_epochs = 120
learning_rate = 3e-4
weight_decay = 5e-4
batch_size = 32
early_stopping_patience = 8
use_focal_loss = True
focal_gamma = 1.5  # 从2.0降低
class_weights = None
dropout_rate = 0.1

# 模型配置
num_layers = 12
embed_dim = 384
```

**预期效果**:
- 测试准确率: 68-72%
- 训练时间: ~6-8小时
- 过拟合差距: 8-12%
- 显存占用: ~4-5GB

---

### 方案C: 快速训练（用于调试）

```python
# main.py 配置
num_epochs = 50
learning_rate = 3e-4
weight_decay = 3e-4
batch_size = 32
early_stopping_patience = 8
use_focal_loss = False  # 关闭Focal Loss
class_weights = None
dropout_rate = 0.1

# 模型配置
num_layers = 12
embed_dim = 384
```

**预期效果**:
- 测试准确率: 65-67%
- 训练时间: ~2.5-3小时
- 过拟合差距: 12-18%

---

## 🔧 高级配置

### 显存优化

**方案A: 大batch训练（适合4060 24GB显存）**
```python
batch_size = 48  # 从32增加到48
gradient_accumulation_steps = 1
```

**预期效果**:
- 显存利用率: 90-95%
- 训练速度: +30-40%
- 显存占用: ~6-7GB

**方案B: 梯度累积（减少显存占用）**
```python
batch_size = 64  # 进一步增加
gradient_accumulation_steps = 2  # 每2个batch更新一次
```

**预期效果**:
- 显存占用: ~8GB
- 训练速度: +50%
- 有效batch size: 128（2×64）

---

### 学习率策略

**方案A: CosineAnnealing（推荐）**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * 10,  # 更长的周期
    eta_min=1e-6,
    last_epoch=-1
)
```

**方案B: OneCycleLR（快速收敛）**
```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)
```

**方案C: ReduceLROnPlateau（自适应）**
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.7,
    patience=8,
    min_lr=1e-6,
    threshold=0.001
)
```

---

## 🎯 推荐配置组合

### 配置1: 稳定高效训练（推荐）

```python
{
    "num_epochs": 100,
    "learning_rate": 3e-4,
    "weight_decay": 3e-4,
    "batch_size": 32,
    "early_stopping_patience": 8,
    "use_focal_loss": True,
    "class_weights": None,
    "dropout_rate": 0.1
}
```

**优点**:
- ✅ 训练时间合理（~5-6小时）
- ✅ 过拟合可控（预期10-15%）
- ✅ 显存占用适中（~4-5GB）
- ✅ 配置简单易调试

**缺点**:
- ⚠️ 可能未达到最优准确率
- ⚠️ 需要后续调参

---

### 配置2: 高性能训练（实验）

```python
{
    "num_epochs": 120,
    "learning_rate": 3e-4,
    "weight_decay": 5e-4,
    "batch_size": 48,
    "early_stopping_patience": 8,
    "use_focal_loss": True,
    "focal_gamma": 1.5,
    "class_weights": None,
    "dropout_rate": 0.1
}
```

**优点**:
- ✅ 更好的准确率（预期68-72%）
- ✅ 更充分训练
- ✅ Focal Loss效果更好发挥

**缺点**:
- ⚠️ 训练时间较长（~6-8小时）
- ⚠️ 显存占用增加（~6-7GB）

---

## 📊 硬件环境适配

### 4060显卡（24GB显存）

**推荐配置**:
- Batch size: 32-48
- 梯度累积: 关闭
- 混合精度: fp16（已在代码中）
- 模型配置: 12层, 384维

**性能估算**:
- 每epoch时间: ~180-220秒
- 总训练时间: 5-8小时
- 峰值显存: 4-5GB

---

## 🚨 硬件问题诊断

### 显存不足（OOM）

**症状**:
- RuntimeError: CUDA out of memory
- 训练中断

**解决方案**:
1. 减小batch size（32→16→8）
2. 减小模型尺寸（num_layers=12→10→8）
3. 关闭混合精度训练（use_mixed_precision=False）
4. 添加梯度累积（gradient_accumulation_steps=2）

---

## 📝 使用说明

### 快速开始

1. 复制配置到main.py
   - 从CONFIG.md中复制所需的配置块
   - 修改Trainer参数

2. 修改训练轮数
   ```python
   num_epochs = 100  # 或120
   ```

3. 开始训练
   ```bash
   python train.py
   ```

---

## 🔍 配置调优建议

### 1. 超参数方向
- 学习率: 3e-4 → 1e-4 → 3e-5（如果过拟合）
- 权重衰减: 3e-4 → 5e-4 → 1e-3（如果过拟合）
- Dropout率: 0.1 → 0.05 → 0.0（如果过拟合）
- Batch size: 32 → 48 → 64（如果显存充足）

### 2. 验证改进
- A/B测试不同配置
- 记录详细指标
- 分析混淆矩阵

---

## 🎯 最佳实践

### 1. 训练前检查
- 数据集完整性
- 数据加载器配置
- 模型参数量
- 显存可用量

### 2. 训练中监控
- 观察loss曲线
- 注意异常值
- 定期保存checkpoint

### 3. 训练后分析
- 查看混淆矩阵
- 分析类别准确率
- 对比不同配置结果

---

**更新日期**: 2026-03-27
**适用版本**: v2.0及以后
**硬件环境**: NVIDIA RTX 4060 (24GB VRAM)
