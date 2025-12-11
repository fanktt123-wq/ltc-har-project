import numpy as np
import torch
import torch.nn.functional as F


def train_eval(model, loader_train, loader_test_clean, loader_test_noisy, device, 
               epochs, lr, use_amp, clip_grad, label_smoothing=0.1, weight_decay=1e-4):
    """训练和评估模型"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    train_accs, test_clean_accs, test_noisy_accs = [], [], []
    train_losses = []
    
    last_h_all = None
    last_input_seq = None
    
    def eval_loader(loader, return_predictions=False):
        """评估数据加载器"""
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

