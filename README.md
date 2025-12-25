# 基于液态时间常数（LTC）的液态神经网络在UCI HAR人体活动识别中的应用

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目实现了一个基于液态时间常数（Liquid Time-Constant, LTC）的液态神经网络（Liquid Neural Network, LNN），用于UCI HAR（Human Activity Recognition）人体活动识别任务。该模型采用连续时间微分方程建模，能够有效处理时序传感器数据。

## 项目概述

本项目构建了一个液态神经网络模型，其核心组件为LTC层，通过微分方程描述神经元状态的动态变化。模型在UCI HAR数据集上进行训练和评估，支持干净测试集和噪声测试集的双重评估，以验证模型的鲁棒性。

### 主要特性

- **连续时间建模**：使用ODE（常微分方程）描述神经元动态
- **液态特性**：时间常数随输入信号动态变化
- **鲁棒性测试**：支持噪声注入和dropout测试
- **丰富可视化**：提供多种可视化工具分析模型行为
- **完整评估**：包含准确率、精确率、召回率、F1分数等指标

## 项目结构

```
lnn/
├── main.py                 # 主程序入口
├── models.py              # LTC模型定义（LTCCell, LTCLayer, LTCHAR）
├── dataset.py             # 数据集加载和处理（UCIHAR类）
├── train.py               # 训练和评估函数
├── metrics.py             # 评估指标计算
├── visualization.py       # 可视化工具
├── config.py              # 配置文件
├── data/                  # 数据目录
│   └── UCI HAR.zip        # UCI HAR数据集压缩包
├── ltc_logs_har/          # 训练日志和输出目录
│   ├── args.txt           # 训练参数记录
│   ├── metrics.txt        # 评估指标结果
│   ├── checkpoint_latest.pth  # 最新模型检查点
├── ltc_har_backup.zip     # 原始单文件版本备份
├── README.md              # 项目说明文档
├── LTC_HAR_实验报告.md     # 实验报告
└── GitHub操作指南.md       # GitHub使用指南
```

## 环境要求

- Python 3.6+
- PyTorch 1.8+
- NumPy
- Matplotlib
- scikit-learn
- TensorBoard

## 安装

### 1. 克隆仓库

```bash
git clone <your-repository-url>
cd lnn
```

### 2. 安装依赖

```bash
pip install torch torchvision matplotlib numpy scikit-learn tensorboard
```

或者使用requirements.txt（如果存在）：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本训练

```bash
python main.py \
    -data-dir ./data \
    -out-dir ./ltc_logs_har \
    -device cuda:0 \
    -epochs 200 \
    -b 256 \
    -hidden 256 \
    -lr 1e-3 \
    -dt 0.05 \
    -dropout 0.3 \
    -label-smoothing 0.1 \
    -weight-decay 1e-4 \
    -amp
```

**注意：** `-data-dir` 参数可以指定为 `./data`，程序会自动从 `data` 文件夹读取数据集。

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-data-dir` | `./data` | UCI HAR数据集目录（程序会自动从data文件夹读取） |
| `-out-dir` | `./ltc_logs_har` | 日志和检查点保存目录 |
| `-device` | `cuda:0` | 设备选择（cuda:0/cpu） |
| `-epochs` | `200` | 训练轮数 |
| `-b` | `256` | 批量大小 |
| `-hidden` | `256` | 隐层大小 |
| `-lr` | `1e-3` | 学习率 |
| `-dt` | `0.05` | LTC细胞的积分步长 |
| `-dropout` | `0.3` | Dropout率 |
| `-label-smoothing` | `0.1` | 标签平滑系数 |
| `-weight-decay` | `1e-4` | 权重衰减（L2正则化） |
| `-amp` | `False` | 是否使用自动混合精度训练 |

## 数据集

本项目使用UCI HAR数据集，包含6种人体活动：

- WALKING（行走）
- WALKING_UPSTAIRS（上楼梯）
- WALKING_DOWNSTAIRS（下楼梯）
- SITTING（坐）
- STANDING（站）
- LAYING（躺）

### 数据准备

**重要：** 请确保数据集zip文件位于项目的 `data` 文件夹中：

```
lnn/
├── data/
│   └── UCI HAR.zip
```

程序会自动从 `data` 文件夹读取zip文件并解压，**不会自动下载**。如果zip文件不存在，程序会报错提示。

首次运行时会自动解压zip文件到 `data/UCI HAR Dataset/` 目录。

## 实验结果

模型在干净测试集和噪声测试集上分别进行评估：

- **干净测试集**：评估模型在理想条件下的性能
- **噪声测试集**：评估模型的鲁棒性

主要评估指标包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-Score）
- 混淆矩阵（Confusion Matrix）

## 可视化

项目提供多种可视化工具，所有可视化结果图片在训练时自动保存到指定的日志目录（默认 `ltc_logs_har/`）中：

- **收敛曲线**：展示训练过程中的准确率和损失变化
- **混淆矩阵**：展示分类混淆情况（干净测试集和噪声测试集）
- **雷达图**：以极坐标形式展示每个类别的精确率、召回率、F1分数
- **指标柱状图**：展示每个类别的详细指标
- **隐状态热力图**：展示神经元状态随时间的变化
- **神经元动力学轨迹**：展示单个神经元的动态行为
- **类别概率流**：展示分类决策的实时变化

所有可视化图片保存在 `-out-dir` 参数指定的目录中（默认为 `ltc_logs_har/`），包括：
- `confusion_matrix_clean.png` / `confusion_matrix_noisy.png` - 混淆矩阵
- `convergence_curves.png` - 收敛曲线
- `radar_chart_clean.png` / `radar_chart_noisy.png` - 雷达图
- `metrics_bar_clean.png` / `metrics_bar_noisy.png` - 指标柱状图
- `hidden_state_heatmap.png` - 隐藏状态热力图
- `class_probability_flow.png` - 类别概率流
- `neuron_dynamics_*.png` - 神经元动力学轨迹图

**展示图片**：项目根目录包含三个展示图片，用于快速预览：
- `confusion_matrix_clean.png` - 混淆矩阵（干净测试集）
- `convergence_curves.png` - 收敛曲线
- `hidden_state_heatmap.png` - 隐藏状态热力图

## 模型架构

### LTC细胞

LTC细胞通过以下微分方程描述神经元状态：

$$\frac{dh}{dt} = -\left(\frac{1}{\tau} + f(x,h)\right) \cdot h + f(x,h) \cdot E_{rev}$$

其中：
- $\tau$ 为基础时间常数
- $f(x,h)$ 为非线性调制函数
- $E_{rev}$ 为反转电位

### 网络结构

- **输入层**：9维传感器信号（3个加速度计 + 3个陀螺仪 + 3个总加速度）
- **LTC层**：LTC细胞组成的连续时间层
- **Dropout层**：防止过拟合
- **输出层**：全连接层，输出6个活动类别

## 团队成员
王健淞-负责模块：
models.py - LTC细胞、LTC层
config.py - 配置参数
周杨硕-负责模块
dataset.py - UCI HAR数据集加载与处理
train.py - 训练循环与评估函数
metrics.py - 评估指标计算
张晓萌-负责模块
main.py - 主程序入口
visualization.py - 可视化工具
README.md - 项目文档
## 参考文献

- Hasani, R., Lechner, M., Amini, A., et al. (2020). Liquid Time-Constant Networks. *arXiv preprint arXiv:2006.04439*.
- Anguita, D., Ghio, A., Oneto, L., et al. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN*.

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

