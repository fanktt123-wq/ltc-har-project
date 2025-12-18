import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LTCCell(nn.Module):
    """LTC (Liquid Time-Constant) Cell"""
    
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.05, 
                 tau_min=1e-3, tau_max=10.0, E_rev_init=0.0):
        """初始化LTC单元：设置输入/隐藏维度、时间步长和可学习参数"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xh = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.E_rev = nn.Parameter(torch.full((hidden_size,), E_rev_init))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数：使用Kaiming初始化权重，均匀初始化偏置，正态初始化时间常数"""
        nn.init.kaiming_uniform_(self.W_hh, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_xh, a=np.sqrt(5))
        nn.init.uniform_(self.bias, -1.0, -0.5)
        nn.init.normal_(self.log_tau, mean=0.0, std=0.1)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """前向传播：通过ODE积分计算当前时间步的隐藏状态"""
        tau = torch.clamp(F.softplus(self.log_tau) + self.tau_min, max=self.tau_max)
        f_val = torch.tanh(F.linear(h_prev, self.W_hh) + F.linear(x_t, self.W_xh) + self.bias)
        dhdt = -(1.0 / tau + f_val) * h_prev + f_val * self.E_rev
        h = h_prev + self.dt * dhdt
        return h


class LTCLayer(nn.Module):
    """LTC Layer，处理序列输入"""
    
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.05):
        """初始化LTC层：创建LTC单元用于处理序列"""
        super().__init__()
        self.cell = LTCCell(input_size, hidden_size, dt)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回所有时间步的隐藏状态和最后一个时间步的隐藏状态"""
        B, T, _ = x.shape
        h = x.new_zeros((B, self.cell.hidden_size))
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


class LTCHAR(nn.Module):
    """LTC模型用于HAR任务"""
    
    def __init__(self, input_size=9, hidden_size=256, num_classes=6, dt=0.05, dropout=0.3):
        """初始化LTC模型：包含LTC层、Dropout和分类头"""
        super().__init__()
        self.ltc = LTCLayer(input_size, hidden_size, dt)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """前向传播，返回分类logits和所有时间步的隐藏状态"""
        h_all, h_last = self.ltc(x)
        h_last = self.dropout(h_last)
        return self.head(h_last), h_all

