# TensorBoard 可视化指南

## 启动 TensorBoard

```powershell
# 方法1: 在项目根目录运行
tensorboard --logdir=./logs --port=6006

# 方法2: 指定端口
tensorboard --logdir=./logs --port=6007

如果以上命令报错，请尝试以下命令强制执行：
python -m tensorboard.main --logdir=./logs
```

然后在浏览器打开：`http://localhost:6006`

---

## TensorBoard 主要内容

### 1. SCALARS（标量）

#### 训练指标
- `Epoch/train_loss` - 训练损失（每个epoch）
- `Epoch/train_accuracy` - 训练准确率（每个epoch）
- `Epoch/test_loss` - 测试损失
- `Epoch/test_accuracy` - 测试准确率

#### Batch 级别详细记录
- `Batch/train_loss` - 每个batch的损失
- `Batch/train_accuracy` - 每个batch的准确率
- `Batch/Class_{EMOTION}_Accuracy` - 每个类别的准确率（每10个batch）
- `Batch/Class_{EMOTION}_Pred_Rate` - 每个类别的预测率（每10个batch）

#### 学习率
- `Learning_Rate` - 学习率变化曲线

#### 时间统计
- `Epoch/Time` - 每个epoch的训练时间
- `Batch/Gradient_Norm` - 梯度范数（每50个batch）

#### HARAMS（超参数）
- 点击 "HPARAMS" 标签页查看实验配置
- 包括：num_epochs, learning_rate, batch_size, use_ls_conv, use_kimi_residual

---

### 2. HISTOGRAMS（直方图）

#### 模型权重分布
- `Weights/encoder.*` - Encoder层权重分布
- `Weights/classifier.*` - 分类器权重分布
- `Gradients/encoder.*` - Encoder层梯度分布

#### 损失分布
- `Epoch/train_loss_distribution` - 训练损失分布
- `Epoch/test_loss_distribution` - 测试损失分布

---

### 3. GRAPHS（图）

- 目前未配置
- 可以添加：PR曲线、ROC曲线等

---

### 4. IMAGES（图像）

#### 混淆矩阵
- `Confusion_Matrix` - 每个epoch的混淆矩阵热力图

---

## 注意：查看 Batch 级别详情

默认配置下，程序**每 10 个 batch 记录一次详细信息**：

- 每个类别的准确率
- 模型权重分布
- 梯度范数

如果需要查看每个 batch，修改 `train.py` 中的 `log_batch_every` 参数：

```python
trainer = Trainer(
    # ... 其他参数
    log_batch_every=1  # 每个batch都记录详细信息
)
```

**注意：记录过多可能会减慢训练速度和占用大量磁盘空间**

---

## TensorBoard 使用技巧

### 1. 对比多个实验

运行不同配置的训练，使用相同的 log 目录结构：

```powershell
# 实验1: Full Model
python train.py  # 使用完整模型

# 实验2: No LSConv
# 修改 train.py 中的 ablation_mode
ablation_mode = {
    'use_ls_conv': False,
    'use_kimi_residual': True
}
```

然后在 TensorBoard 中点击左上角的实验名称进行对比。

### 2. 下载图表

- 右键图表 → "Download" → PNG/SVG

### 3. 刷新数据

- 点击页面左上角的刷新按钮（或按 Ctrl+R）

### 4. 查看特定范围

- 在图表左侧可以调整显示的步数范围
- 滚轮到想要查看的 epoch 范围

---

## 常见问题

### Q: TensorBoard 端口被占用
```powershell
# 方案1: 换个端口
tensorboard --logdir=./logs --port=6007

# 方案2: 杀死占用进程
taskkill /F /IM tensorboard.exe
```

### Q: 查看历史实验
```powershell
# 合并多个日志目录
tensorboard --logdir_spec=logs1:logs2:logs3
```

### Q: 远程访问
```powershell
# 允许远程访问
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

---

## 推荐查看顺序

### 训练初期（Epoch 1-10）
1. **SCALARS > Epoch/train_loss** - 检查损失是否下降
2. **SCALARS > Learning_Rate** - 查看学习率调度
3. **HISTOGRAMS > Weights/encoder** - 检查权重初始化

### 训练中期（Epoch 11-30）
1. **SCALARS > Epoch/train_accuracy** - 准确率是否稳定提升
2. **SCALARS > Batch/train_loss** - batch级别损失波动
3. **IMAGES > Confusion_Matrix** - 查看混淆矩阵变化

### 训练后期（Epoch 31-50）
1. **SCALARS > Epoch/test_accuracy** - 测试集准确率趋势
2. **HISTOGRAMS > Gradients/encoder** - 梯度是否稳定
3. **HPARAMS** - 确认超参数配置

---

## 导出报告

训练完成后，在 TensorBoard 中可以：

1. **导出图表**: 右键 "Download as PNG"
2. **复制数据**: 鼠标悬停在点上查看具体数值
3. **截图保存**: 使用系统截图工具

---

## 脚本位置

- 训练脚本: `h:\vit mixed with lsnet\train.py`
- 日志目录: `./logs/`
- 模型保存: `./checkpoints/`
- 注意力图: `./logs/attention_maps/`
- 混淆矩阵: `./logs/confusion_matrices/`
- Batch日志: `./logs/batch_logs/`

---

**祝训练顺利！🚀**
