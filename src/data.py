"""MNIST veri setini yükler, train/validation/test olarak ayırır ve normalize eder.

Ayrım:
    MNIST'in resmi 60.000'lik eğitim kısmı -> 50.000 train + 10.000 validation
    MNIST'in resmi 10.000'lik test kısmı   -> test (yalnızca en sonda, bir kez kullanılır)
"""
from pathlib import Path

import torch
from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VAL_SIZE = 10_000
SPLIT_SEED = 42  # ayrım her çalıştırmada birebir aynı olsun diye sabit


def load_raw():
    """MNIST'i indirir (yalnızca ilk seferde) ve ham uint8 tensörler olarak döner."""
    train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)
    return train.data, train.targets, test.data, test.targets


def split_indices(n, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """0..n-1 indekslerini karıştırıp train ve validation olarak ikiye böler."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    return perm[val_size:], perm[:val_size]


def get_splits(include_test=False):
    """Ham (uint8, 28x28) parçaları döner.

    Test seti varsayılan olarak döndürülmez: model geliştirirken yanlışlıkla
    kullanılmasın diye yalnızca açıkça istendiğinde (include_test=True) verilir.
    """
    x_full, y_full, x_test, y_test = load_raw()
    train_idx, val_idx = split_indices(len(x_full))
    splits = {
        "train": (x_full[train_idx], y_full[train_idx]),
        "val": (x_full[val_idx], y_full[val_idx]),
    }
    if include_test:
        splits["test"] = (x_test, y_test)
    return splits


def compute_mean_std(x_uint8):
    """[0, 1] ölçeğine getirilmiş piksellerin ortalaması ve standart sapması."""
    x = x_uint8.float() / 255.0
    return x.mean().item(), x.std().item()


def load_data(normalization="standard", include_test=False, device="cpu"):
    """Modele girecek hâle getirilmiş veriyi ve normalizasyon istatistiklerini döner.

    normalization:
        "none"     -> ham pikseller, [0, 255]
        "minmax"   -> [0, 1] aralığı
        "standard" -> (x - ortalama) / std; ortalama ve std YALNIZCA train'den hesaplanır
    Her görüntü 28x28'den 784 uzunluğunda bir vektöre düzleştirilir (MLP girdisi).
    """
    splits = get_splits(include_test)
    mean, std = compute_mean_std(splits["train"][0])

    def transform(x):
        x = x.float().reshape(len(x), -1)  # (N, 28, 28) -> (N, 784)
        if normalization == "none":
            return x
        x = x / 255.0
        if normalization == "minmax":
            return x
        if normalization == "standard":
            return (x - mean) / std
        raise ValueError(f"Bilinmeyen normalizasyon: {normalization}")

    data = {name: (transform(x).to(device), y.to(device)) for name, (x, y) in splits.items()}
    return data, (mean, std)
