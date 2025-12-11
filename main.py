import os
import sys
import argparse
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from models import LTCHAR
from train import train_eval
from metrics import compute_metrics
from visualization import (
    plot_confusion_matrix, plot_radar_chart, plot_convergence_curves,
    plot_metrics_bar, plot_batch_size_line, plot_hidden_state_heatmap,
    plot_class_probability_flow, plot_neuron_dynamics_trace
)
from config import CLASS_NAMES, NUM_CLASSES


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
        args.data_dir, args.b, args.j, args.drop, args.noise, 
        apply_noise_to_train=True, apply_noise_to_test=False
    )
    _, test_loader_noisy = get_dataloaders(
        args.data_dir, args.b, args.j, args.drop_noisy, args.noise_noisy, 
        apply_noise_to_train=False, apply_noise_to_test=True
    )

    print(f"Using device={device}, epochs={args.epochs}, batch={args.b}, workers={args.j}, hidden={args.hidden}")
    print(f"Train: drop={args.drop}, noise={args.noise} | Test clean: drop=0, noise=0 | Test noisy: drop={args.drop_noisy}, noise={args.noise_noisy}")
    print(f"Regularization: dropout={args.dropout}, label_smoothing={args.label_smoothing}, weight_decay={args.weight_decay}")

    model = LTCHAR(input_size=9, hidden_size=args.hidden, num_classes=NUM_CLASSES, dt=args.dt, dropout=args.dropout)

    writer = SummaryWriter(args.out_dir)

    results = train_eval(
        model, train_loader, test_loader_clean, test_loader_noisy, device, 
        args.epochs, args.lr, use_amp, clip_grad,
        label_smoothing=args.label_smoothing, weight_decay=args.weight_decay
    )
    
    train_accs = results['train_accs']
    test_clean_accs = results['test_clean_accs']
    test_noisy_accs = results['test_noisy_accs']
    train_losses = results['train_losses']
    
    print("\n=== Computing Detailed Metrics ===")
    metrics_clean = compute_metrics(results['y_true_clean'], results['y_pred_clean'], NUM_CLASSES)
    metrics_noisy = compute_metrics(results['y_true_noisy'], results['y_pred_noisy'], NUM_CLASSES)
    
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
    
    # 生成可视化
    plot_confusion_matrix(metrics_clean['confusion_matrix'], CLASS_NAMES, 
                         os.path.join(args.out_dir, "confusion_matrix_clean.png"), 
                         "Confusion Matrix (Clean Test)")
    plot_confusion_matrix(metrics_noisy['confusion_matrix'], CLASS_NAMES, 
                         os.path.join(args.out_dir, "confusion_matrix_noisy.png"), 
                         "Confusion Matrix (Noisy Test)")
    
    plot_radar_chart(metrics_clean, CLASS_NAMES, os.path.join(args.out_dir, "radar_chart_clean.png"))
    plot_radar_chart(metrics_noisy, CLASS_NAMES, os.path.join(args.out_dir, "radar_chart_noisy.png"))
    
    plot_convergence_curves(train_accs, test_clean_accs, test_noisy_accs, train_losses,
                           os.path.join(args.out_dir, "convergence_curves.png"))
    
    plot_metrics_bar(metrics_clean, CLASS_NAMES, os.path.join(args.out_dir, "metrics_bar_clean.png"))
    plot_metrics_bar(metrics_noisy, CLASS_NAMES, os.path.join(args.out_dir, "metrics_bar_noisy.png"))
    
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
            plot_class_probability_flow(logits_seq, CLASS_NAMES,
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

