import os
import zipfile
import urllib.request
import numpy as np
import torch
import torch.utils.data as data


UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def download_and_extract(root: str):
    """下载并解压UCI HAR数据集"""
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
    """加载训练或测试集数据"""
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
    """UCI HAR数据集类"""
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
                X = (X - self.train_mean) / self.train_std
            else:
                train_X, _ = load_split(extract_dir, "train")
                UCIHAR.train_mean = train_X.mean(axis=(0, 1), keepdims=True)
                UCIHAR.train_std = train_X.std(axis=(0, 1), keepdims=True) + 1e-8
                X = (X - self.train_mean) / self.train_std
        
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


def get_dataloaders(data_dir, batch_size, workers, drop_ratio, noise_sigma, 
                    apply_noise_to_train=True, apply_noise_to_test=True):
    """获取训练和测试数据加载器"""
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
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=workers, pin_memory=True, drop_last=True
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=workers, pin_memory=True
    )
    return train_loader, test_loader

