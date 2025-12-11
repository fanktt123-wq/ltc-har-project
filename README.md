# 基于液态时间常数（LTC）的液态神经网络在UCI HAR人体活动识别中的应用

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目实现了一个基于液态时间常数（Liquid Time-Constant, LTC）的液态神经网络（Liquid Neural Network, LNN），用于UCI HAR（Human Activity Recognition）人体活动识别任务。该模型采用连续时间微分方程建模，能够有效处理时序传感器数据。

## 📋 项目概述

本项目构建了一个液态神经网络模型，其核心组件为LTC层，通过微分方程描述神经元状态的动态变化。模型在UCI HAR数据集上进行训练和评估，支持干净测试集和噪声测试集的双重评估，以验证模型的鲁棒性。

### 主要特性

- ✅ **连续时间建模**：使用ODE（常微分方程）描述神经元动态
- ✅ **液态特性**：时间常数随输入信号动态变化
- ✅ **鲁棒性测试**：支持噪声注入和dropout测试
- ✅ **丰富可视化**：提供多种可视化工具分析模型行为
- ✅ **完整评估**：包含准确率、精确率、召回率、F1分数等指标

## 🏗️ 项目结构

```
lnn/
├── main.py                 # 主程序入口
├── ltc_har.py             # LTC模型实现和训练脚本
├── models.py              # 模型定义
├── dataset.py             # 数据集加载
├── train.py               # 训练和评估函数
├── metrics.py             # 评估指标计算
├── visualization.py       # 可视化工具
├── config.py              # 配置文件
├── data/                  # 数据目录
├── log/                   # 日志目录
└── README.md              # 项目说明文档
```

## 🔧 环境要求

- Python 3.6+
- PyTorch 1.8+
- NumPy
- Matplotlib
- scikit-learn
- TensorBoard

## 📦 安装

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

## 🚀 使用方法

### 基本训练

```bash
python main.py \
    -data-dir ./har_data \
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

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-data-dir` | `./har_data` | UCI HAR数据集根目录 |
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

## 📊 数据集

本项目使用UCI HAR数据集，包含6种人体活动：

- WALKING（行走）
- WALKING_UPSTAIRS（上楼梯）
- WALKING_DOWNSTAIRS（下楼梯）
- SITTING（坐）
- STANDING（站）
- LAYING（躺）

数据集会在首次运行时自动下载。

## 🧪 实验结果

模型在干净测试集和噪声测试集上分别进行评估：

- **干净测试集**：评估模型在理想条件下的性能
- **噪声测试集**：评估模型的鲁棒性

主要评估指标包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-Score）
- 混淆矩阵（Confusion Matrix）

## 📈 可视化

项目提供多种可视化工具：

- **收敛曲线**：展示训练过程中的准确率和损失变化
- **混淆矩阵**：展示分类混淆情况
- **指标柱状图**：展示每个类别的详细指标
- **隐状态热力图**：展示神经元状态随时间的变化
- **神经元动力学轨迹**：展示单个神经元的动态行为
- **类别概率流**：展示分类决策的实时变化

## 🔬 模型架构

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

## 👥 团队成员

<!-- 在这里添加团队成员和分工 -->

**项目负责人：** [组长姓名]

**团队成员：**
- [成员1姓名] - [负责模块/任务]
- [成员2姓名] - [负责模块/任务]
- [成员3姓名] - [负责模块/任务]

## 📝 参考文献

- Hasani, R., Lechner, M., Amini, A., et al. (2020). Liquid Time-Constant Networks. *arXiv preprint arXiv:2006.04439*.
- Anguita, D., Ghio, A., Oneto, L., et al. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN*.

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢UCI机器学习仓库提供HAR数据集。

---

**注意：** 本项目为学术研究用途，仅供学习和参考。

