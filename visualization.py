import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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
    """绘制隐藏状态热力图"""
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
    """绘制类别概率流"""
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
    """绘制神经元动力学轨迹"""
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

