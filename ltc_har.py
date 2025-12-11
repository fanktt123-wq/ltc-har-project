import os
import sys
import argparse
import time
import zipfile
import urllib.request
from typing import Tuple, List, Dict
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


# ----------------------------
# Dataset (UCI HAR)
# ----------------------------
def download_and_extract(root: str):
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "uci_har.zip")
    extract_dir = os.path.join(root, "UCI HAR Dataset")
    if os.path.isdir(extract_dir):
        return extract_dir
    if not os.path.exists(zip_path):
        print("Downloading UCI HAR...")
        urllib.request.urlretrieve(UCI_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return extract_dir


def load_split(extract_dir: str, split: str):
    assert split in ["train", "test"]
    inertial_dir = os.path.join(extract_dir, split, "Inertial Signals")
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    signals = []
    for name in signal_names:
        path = os.path.join(inertial_dir, f"{name}_{split}.txt")
        sig = np.loadtxt(path)
        signals.append(sig)
    X = np.stack(signals, axis=-1)
    y = np.loadtxt(os.path.join(extract_dir, split, f"y_{split}.txt")) - 1
    return X.astype(np.float32), y.astype(np.int64)


class UCIHAR(data.Dataset):
    train_mean = None
    train_std = None
    
    def __init__(self, root: str, split: str, drop_ratio: float = 0.0, noise_sigma: float = 0.0):
        extract_dir = download_and_extract(root)
        X, y = load_split(extract_dir, split)
        
        if split == "train":
            self.train_mean = X.mean(axis=(0, 1), keepdims=True)
            self.train_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
            UCIHAR.train_mean = self.train_mean
            UCIHAR.train_std = self.train_std
            X = (X - self.train_mean) / self.train_std
        else:
            if UCIHAR.train_mean is not None:
                X = (X - UCIHAR.train_mean) / UCIHAR.train_std
            else:
                train_X, _ = load_split(extract_dir, "train")
                UCIHAR.train_mean = train_X.mean(axis=(0, 1), keepdims=True)
                UCIHAR.train_std = train_X.std(axis=(0, 1), keepdims=True) + 1e-8
                X = (X - UCIHAR.train_mean) / UCIHAR.train_std
        
        self.X = X
        self.y = y
        self.drop_ratio = drop_ratio
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.X[idx]
        if self.drop_ratio > 0:
            keep = np.random.binomial(1, 1 - self.drop_ratio, size=(seq.shape[0], 1)).astype(np.float32)
            seq = seq * keep
        if self.noise_sigma > 0:
            seq = seq + np.random.normal(0, self.noise_sigma, size=seq.shape).astype(np.float32)
        seq = seq.astype(np.float32, copy=False)
        return torch.from_numpy(seq), torch.tensor(self.y[idx], dtype=torch.long)


def get_dataloaders(data_dir, batch_size, workers, drop_ratio, noise_sigma, apply_noise_to_train=True, apply_noise_to_test=True):
    train_set = UCIHAR(
        data_dir,
        split="train",
        drop_ratio=drop_ratio if apply_noise_to_train else 0.0,
        noise_sigma=noise_sigma if apply_noise_to_train else 0.0,
    )
    test_set = UCIHAR(
        data_dir,
        split="test",
        drop_ratio=drop_ratio if apply_noise_to_test else 0.0,
        noise_sigma=noise_sigma if apply_noise_to_test else 0.0,
    )
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------
# LTC Cell
# ----------------------------
class LTCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.05, 
                 tau_min=1e-3, tau_max=10.0, E_rev_init=0.0):
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
        nn.init.kaiming_uniform_(self.W_hh, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_xh, a=np.sqrt(5))
        nn.init.uniform_(self.bias, -1.0, -0.5)
        nn.init.normal_(self.log_tau, mean=0.0, std=0.1)


    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(F.softplus(self.log_tau) + self.tau_min, max=self.tau_max)
        f_val = torch.tanh(F.linear(h_prev, self.W_hh) + F.linear(x_t, self.W_xh) + self.bias)
        dhdt = -(1.0 / tau + f_val) * h_prev + f_val * self.E_rev
        h = h_prev + self.dt * dhdt
        return h


class LTCLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.05):
        super().__init__()
        self.cell = LTCCell(input_size, hidden_size, dt)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        h = x.new_zeros((B, self.cell.hidden_size))
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


class LTCHAR(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, num_classes=6, dt=0.05, dropout=0.3):
        super().__init__()
        self.ltc = LTCLayer(input_size, hidden_size, dt)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_all, h_last = self.ltc(x)
        h_last = self.dropout(h_last)
        return self.head(h_last), h_all


# ----------------------------
# Training / Eval
# ----------------------------
def train_eval(model, loader_train, loader_test_clean, loader_test_noisy, device, epochs, lr, use_amp, clip_grad, label_smoothing=0.1, weight_decay=1e-4):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    train_accs, test_clean_accs, test_noisy_accs = [], [], []
    train_losses = []
    
    last_h_all = None
    last_input_seq = None
    
    for epoch in range(epochs):
        model.train()
        tot, corr = 0, 0
        epoch_loss = 0.0
        for seq, label in loader_train:
            seq, label = seq.to(device, dtype=torch.float32), label.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, h_all = model(seq)
                    loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
                scaler.scale(loss).backward()
                if clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits, h_all = model(seq)
                loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            tot += label.numel()
            corr += (logits.argmax(1) == label).sum().item()
            epoch_loss += loss.item() * label.numel()
        train_accs.append(corr / tot)
        train_losses.append(epoch_loss / tot)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            def eval_loader(loader, return_predictions=False):
                tot_t, corr_t = 0, 0
                all_preds, all_labels = [], []
                last_h = None
                last_seq = None
                for seq, label in loader:
                    seq, label = seq.to(device, dtype=torch.float32), label.to(device)
                    logits, h_all = model(seq)
                    preds = logits.argmax(1)
                    tot_t += label.numel()
                    corr_t += (preds == label).sum().item()
                    if return_predictions:
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(label.cpu().numpy())
                    last_h = h_all
                    last_seq = seq
                if return_predictions:
                    return corr_t / tot_t, np.array(all_preds), np.array(all_labels), last_h, last_seq
                return corr_t / tot_t
            
            acc_clean = eval_loader(loader_test_clean)
            acc_noisy = eval_loader(loader_test_noisy)
            test_clean_accs.append(acc_clean)
            test_noisy_accs.append(acc_noisy)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: train_acc={train_accs[-1]:.4f} train_loss={train_losses[-1]:.4f} "
                f"clean_acc={test_clean_accs[-1]:.4f} noisy_acc={test_noisy_accs[-1]:.4f}"
            )
    
    model.eval()
    with torch.no_grad():
        _, y_pred_clean, y_true_clean, last_h_all, last_input_seq = eval_loader(loader_test_clean, return_predictions=True)
        _, y_pred_noisy, y_true_noisy, _, _ = eval_loader(loader_test_noisy, return_predictions=True)
    
    return {
        'train_accs': train_accs,
        'test_clean_accs': test_clean_accs,
        'test_noisy_accs': test_noisy_accs,
        'train_losses': train_losses,
        'last_h_all': last_h_all,
        'last_input_seq': last_input_seq,
        'y_pred_clean': y_pred_clean,
        'y_true_clean': y_true_clean,
        'y_pred_noisy': y_pred_noisy,
        'y_true_noisy': y_true_noisy,
    }


# ----------------------------
# Visualization Functions
# ----------------------------
def compute_metrics(y_true, y_pred, num_classes=6):
    """计算准确率、精确率、召回率、F1分数"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': acc,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    }


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel='True Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_radar_chart(metrics_dict, class_names, save_path):
    """绘制雷达图：展示每个类别的精确率、召回率、F1"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), dpi=150)
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    precision = metrics_dict['precision_per_class'].tolist()
    recall = metrics_dict['recall_per_class'].tolist()
    f1 = metrics_dict['f1_per_class'].tolist()
    
    precision += precision[:1]
    recall += recall[:1]
    f1 += f1[:1]
    
    ax.plot(angles, precision, 'o-', linewidth=2, label='Precision', color='blue')
    ax.fill(angles, precision, alpha=0.25, color='blue')
    ax.plot(angles, recall, 'o-', linewidth=2, label='Recall', color='green')
    ax.fill(angles, recall, alpha=0.25, color='green')
    ax.plot(angles, f1, 'o-', linewidth=2, label='F1-Score', color='red')
    ax.fill(angles, f1, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart (Per-Class Metrics)', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_curves(train_accs, test_clean_accs, test_noisy_accs, train_losses, save_path):
    """绘制准确率和损失率的收敛曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    epochs = range(len(train_accs))
    ax1.plot(epochs, train_accs, label='Train Accuracy', color='blue', linewidth=2)
    ax1.plot(epochs, test_clean_accs, label='Test Clean Accuracy', color='green', linewidth=2)
    ax1.plot(epochs, test_noisy_accs, label='Test Noisy Accuracy', color='red', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Convergence', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if train_losses:
        ax2.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Loss Convergence', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_bar(metrics_dict, class_names, save_path):
    """绘制每个类别的精确率柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = metrics_dict['precision_per_class']
    recall = metrics_dict['recall_per_class']
    f1 = metrics_dict['f1_per_class']
    
    ax.bar(x - width, precision, width, label='Precision', color='blue', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='green', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', color='red', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_batch_size_line(batch_results, save_path):
    """绘制不同batch size的性能折线图"""
    if not batch_results:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    batch_sizes = sorted(batch_results.keys())
    accs = [batch_results[bs]['accuracy'] for bs in batch_sizes]
    f1s = [batch_results[bs]['f1_macro'] for bs in batch_sizes]
    
    ax.plot(batch_sizes, accs, 'o-', label='Accuracy', linewidth=2, markersize=8, color='blue')
    ax.plot(batch_sizes, f1s, 's-', label='F1-Score (Macro)', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance vs Batch Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hidden_state_heatmap(h_all, save_path, title="Hidden State Heatmap"):
    h_sample = h_all[0].detach().cpu().numpy()
    h_sample = h_sample.T
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    im = ax.imshow(h_sample, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hidden State Value', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_probability_flow(logits_seq, class_names, save_path, title="Class Probability Flow", temperature=2.0):
    scaled_logits = logits_seq / temperature
    probs = F.softmax(scaled_logits, dim=-1).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    time_steps = np.arange(probs.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        ax.plot(time_steps, probs[:, i], label=name, linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_neuron_dynamics_trace(h_trace, input_seq, neuron_idx, save_path, title="Neuron Dynamics Trace"):
    h_neuron = h_trace[:, neuron_idx].detach().cpu().numpy()
    if input_seq.dim() == 3:
        input_feat = input_seq[0, :, 0].detach().cpu().numpy()
    else:
        input_feat = input_seq[:, 0].detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)
    time_steps = np.arange(len(h_neuron))
    
    ax1.plot(time_steps, h_neuron, label=f'Neuron {neuron_idx} Hidden State', linewidth=2, color='blue')
    ax1.set_ylabel('Hidden State Value', fontsize=12)
    ax1.set_title(f'{title} - Neuron {neuron_idx}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_steps, input_feat, label='Input Signal (Feature 0)', linewidth=2, color='green', alpha=0.7)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Input Value', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="LTC (closer-to-paper) on UCI HAR with clean/noisy test eval")
    parser.add_argument("-data-dir", type=str, default="./har_data", help="root dir for UCI HAR")
    parser.add_argument("-out-dir", type=str, default="./ltc_logs_har", help="dir for logs/checkpoints")
    parser.add_argument("-device", type=str, default="cuda:0", help="device")
    parser.add_argument("-epochs", type=int, default=200, help="training epochs")
    parser.add_argument("-b", type=int, default=256, help="batch size")
    parser.add_argument("-j", type=int, default=4, help="data loader workers")
    parser.add_argument("-hidden", type=int, default=256, help="hidden size")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-dt", type=float, default=0.05, help="integration step for LTC cell")
    parser.add_argument("-drop", type=float, default=0.0, help="drop ratio for training")
    parser.add_argument("-noise", type=float, default=0.0, help="noise sigma for training")
    parser.add_argument("-drop-noisy", type=float, default=0.5, help="drop ratio for noisy test")
    parser.add_argument("-noise-noisy", type=float, default=0.0, help="noise sigma for noisy test")
    parser.add_argument("-amp", action="store_true", help="use AMP on CUDA")
    parser.add_argument("-no-clip", action="store_true", help="disable grad clipping")
    parser.add_argument("-dropout", type=float, default=0.3, help="dropout rate")
    parser.add_argument("-label-smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument("-weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("-temperature", type=float, default=2.0, help="temperature scaling for probability visualization")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.txt"), "w", encoding="utf-8") as f:
        f.write(str(args))
        f.write("\n" + " ".join(sys.argv))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    clip_grad = not args.no_clip

    train_loader, test_loader_clean = get_dataloaders(
        args.data_dir, args.b, args.j, args.drop, args.noise, apply_noise_to_train=True, apply_noise_to_test=False
    )
    _, test_loader_noisy = get_dataloaders(
        args.data_dir, args.b, args.j, args.drop_noisy, args.noise_noisy, apply_noise_to_train=False, apply_noise_to_test=True
    )

    print(f"Using device={device}, epochs={args.epochs}, batch={args.b}, workers={args.j}, hidden={args.hidden}")
    print(f"Train: drop={args.drop}, noise={args.noise} | Test clean: drop=0, noise=0 | Test noisy: drop={args.drop_noisy}, noise={args.noise_noisy}")
    print(f"Regularization: dropout={args.dropout}, label_smoothing={args.label_smoothing}, weight_decay={args.weight_decay}")

    model = LTCHAR(input_size=9, hidden_size=args.hidden, num_classes=6, dt=args.dt, dropout=args.dropout)

    writer = SummaryWriter(args.out_dir)

    results = train_eval(
        model, train_loader, test_loader_clean, test_loader_noisy, device, args.epochs, args.lr, use_amp, clip_grad,
        label_smoothing=args.label_smoothing, weight_decay=args.weight_decay
    )
    
    train_accs = results['train_accs']
    test_clean_accs = results['test_clean_accs']
    test_noisy_accs = results['test_noisy_accs']
    train_losses = results['train_losses']
    
    class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    num_classes = 6
    
    print("\n=== Computing Detailed Metrics ===")
    metrics_clean = compute_metrics(results['y_true_clean'], results['y_pred_clean'], num_classes)
    metrics_noisy = compute_metrics(results['y_true_noisy'], results['y_pred_noisy'], num_classes)
    
    print(f"Clean Test - Accuracy: {metrics_clean['accuracy']:.4f}, "
          f"Precision (Macro): {metrics_clean['precision_macro']:.4f}, "
          f"Recall (Macro): {metrics_clean['recall_macro']:.4f}, "
          f"F1 (Macro): {metrics_clean['f1_macro']:.4f}")
    print(f"Noisy Test - Accuracy: {metrics_noisy['accuracy']:.4f}, "
          f"Precision (Macro): {metrics_noisy['precision_macro']:.4f}, "
          f"Recall (Macro): {metrics_noisy['recall_macro']:.4f}, "
          f"F1 (Macro): {metrics_noisy['f1_macro']:.4f}")
    
    print("\n=== Computing Inference Time ===")
    model.eval()
    with torch.no_grad():
        test_seq, test_label = next(iter(test_loader_clean))
        test_seq = test_seq.to(device, dtype=torch.float32)
        for _ in range(10):
            _ = model(test_seq)[0]
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        for _ in range(100):
            _ = model(test_seq)[0]
        torch.cuda.synchronize() if device.type == "cuda" else None
        inference_time = (time.time() - start_time) / 100.0
        print(f"Average inference time per sample: {inference_time*1000:.2f} ms")
    
    metrics_file = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("=== Clean Test Set Metrics ===\n")
        f.write(f"Accuracy: {metrics_clean['accuracy']:.4f}\n")
        f.write(f"Precision (Macro): {metrics_clean['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro): {metrics_clean['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro): {metrics_clean['f1_macro']:.4f}\n")
        f.write(f"\nPer-Class Precision: {metrics_clean['precision_per_class']}\n")
        f.write(f"Per-Class Recall: {metrics_clean['recall_per_class']}\n")
        f.write(f"Per-Class F1: {metrics_clean['f1_per_class']}\n")
        f.write(f"\n=== Noisy Test Set Metrics ===\n")
        f.write(f"Accuracy: {metrics_noisy['accuracy']:.4f}\n")
        f.write(f"Precision (Macro): {metrics_noisy['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro): {metrics_noisy['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro): {metrics_noisy['f1_macro']:.4f}\n")
        f.write(f"\nPer-Class Precision: {metrics_noisy['precision_per_class']}\n")
        f.write(f"Per-Class Recall: {metrics_noisy['recall_per_class']}\n")
        f.write(f"Per-Class F1: {metrics_noisy['f1_per_class']}\n")
        f.write(f"\n=== Inference Time ===\n")
        f.write(f"Average inference time per sample: {inference_time*1000:.2f} ms\n")
    
    plot_confusion_matrix(metrics_clean['confusion_matrix'], class_names, 
                         os.path.join(args.out_dir, "confusion_matrix_clean.png"), 
                         "Confusion Matrix (Clean Test)")
    plot_confusion_matrix(metrics_noisy['confusion_matrix'], class_names, 
                         os.path.join(args.out_dir, "confusion_matrix_noisy.png"), 
                         "Confusion Matrix (Noisy Test)")
    
    plot_radar_chart(metrics_clean, class_names, os.path.join(args.out_dir, "radar_chart_clean.png"))
    plot_radar_chart(metrics_noisy, class_names, os.path.join(args.out_dir, "radar_chart_noisy.png"))
    
    plot_convergence_curves(train_accs, test_clean_accs, test_noisy_accs, train_losses,
                           os.path.join(args.out_dir, "convergence_curves.png"))
    
    plot_metrics_bar(metrics_clean, class_names, os.path.join(args.out_dir, "metrics_bar_clean.png"))
    plot_metrics_bar(metrics_noisy, class_names, os.path.join(args.out_dir, "metrics_bar_noisy.png"))
    
    batch_results = {args.b: metrics_clean}
    plot_batch_size_line(batch_results, os.path.join(args.out_dir, "batch_size_line.png"))
    
    if results['last_h_all'] is not None and results['last_input_seq'] is not None:
        print("\n=== Generating Neuroscience Visualization ===")
        plot_hidden_state_heatmap(results['last_h_all'], 
                                 os.path.join(args.out_dir, "hidden_state_heatmap.png"),
                                 "Hidden State Heatmap (LTC - Continuous Flow)")
        
        model.eval()
        with torch.no_grad():
            sample_seq = results['last_input_seq'][:1]
            sample_h_all = results['last_h_all'][:1]
            logits_seq = []
            for t in range(sample_h_all.shape[1]):
                h_t = sample_h_all[:, t, :]
                logits_t = model.head(h_t)
                logits_seq.append(logits_t)
            logits_seq = torch.cat(logits_seq, dim=0)
            plot_class_probability_flow(logits_seq, class_names,
                                      os.path.join(args.out_dir, "class_probability_flow.png"),
                                      "Class Probability Flow (LTC Real-time Decision)",
                                      temperature=args.temperature)
        
        for neuron_idx in [0, args.hidden//4, args.hidden//2, args.hidden*3//4]:
            plot_neuron_dynamics_trace(sample_h_all[0], sample_seq[0], neuron_idx,
                                      os.path.join(args.out_dir, f"neuron_dynamics_{neuron_idx}.png"),
                                      "Neuron Dynamics Trace (ODE Integration)")
    
    writer.add_hparams(
        vars(args),
        {
            "final_train_acc": train_accs[-1],
            "final_clean_acc": test_clean_accs[-1],
            "final_noisy_acc": test_noisy_accs[-1],
            "clean_f1_macro": metrics_clean['f1_macro'],
            "noisy_f1_macro": metrics_noisy['f1_macro'],
            "inference_time_ms": inference_time * 1000,
        },
    )
    writer.close()

    torch.save({"model": model.state_dict()}, os.path.join(args.out_dir, "checkpoint_latest.pth"))
    print(f"\nDone. All outputs saved to {args.out_dir}")
    print("Generated visualizations:")
    print("  - Metrics: metrics.txt")
    print("  - Confusion matrices: confusion_matrix_clean.png, confusion_matrix_noisy.png")
    print("  - Radar charts: radar_chart_clean.png, radar_chart_noisy.png")
    print("  - Convergence curves: convergence_curves.png")
    print("  - Metrics bars: metrics_bar_clean.png, metrics_bar_noisy.png")
    print("  - Hidden state heatmap: hidden_state_heatmap.png")
    print("  - Class probability flow: class_probability_flow.png")
    print("  - Neuron dynamics traces: neuron_dynamics_*.png")


if __name__ == "__main__":
    main()

