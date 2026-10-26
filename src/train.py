"""Eğitim döngüsü: bir yapılandırmayı (config) alır, modeli eğitir ve her epoch'un metriklerini kaydeder."""
import copy
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

from model import MLP, count_parameters

# Deney 1 (baseline) ayarları. Diğer deneyler yalnızca değiştirdikleri anahtarları verir.
DEFAULT_CONFIG = {
    "hidden": (256, 128),
    "activation": "relu",
    "dropout": 0.0,
    "batchnorm": False,
    "loss": "cross_entropy",
    "optimizer": "adam",
    "lr": 1e-3,
    "momentum": 0.9,      # yalnızca optimizer="sgd_momentum" için
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 20,
    "patience": None,     # early stopping: val loss bu kadar epoch iyileşmezse dur (None = kapalı)
    "seed": 0,
}


def make_loss_fn(name):
    """Modelin ham çıktısını (logit) ve gerçek etiketleri alıp tek bir loss değeri döndüren fonksiyon."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()  # içinde softmax + negatif log olasılık var
    if name == "mse":
        # Softmax olasılıkları ile one-hot etiket arasındaki ortalama karesel fark
        return lambda logits, y: F.mse_loss(F.softmax(logits, dim=1), F.one_hot(y, logits.shape[1]).float())
    if name == "multi_margin":
        return nn.MultiMarginLoss()  # SVM'deki hinge loss'un çok sınıflı hâli
    raise ValueError(f"Bilinmeyen loss: {name}")


def make_optimizer(cfg, params):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd_momentum":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"Bilinmeyen optimizer: {cfg['optimizer']}")


def sync(device):
    """GPU asenkron çalışır; süre ölçmeden önce kuyruktaki işlerin bitmesini bekle."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


_warmed_up = set()


def warm_up(device):
    """İlk GPU çağrısındaki tek seferlik kütüphane yükleme maliyetini süre ölçümünün dışında tutar.

    Bu yapılmazsa programdaki ilk eğitimin ilk epoch'u ~5 sn, sonrakiler ~0.7 sn ölçülüyor.
    """
    if str(device) in _warmed_up:
        return
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(64, 784, device=device)
    y = torch.zeros(64, dtype=torch.long, device=device)
    nn.functional.cross_entropy(model(x), y).backward()
    optimizer.step()
    sync(device)
    _warmed_up.add(str(device))


def train_one_epoch(model, x, y, loss_fn, optimizer, batch_size, generator):
    """Train verisinin tamamı üzerinden bir kez geçer (bir epoch)."""
    model.train()  # dropout / batchnorm eğitim davranışında
    perm = torch.randperm(len(x), device=x.device, generator=generator)  # her epoch farklı sıra
    for i in range(0, len(x), batch_size):
        idx = perm[i:i + batch_size]
        logits = model(x[idx])           # 1) forward pass: tahmin
        loss = loss_fn(logits, y[idx])   # 2) loss: ne kadar yanlış?
        optimizer.zero_grad()            # 3) önceki adımın gradyanlarını sıfırla
        loss.backward()                  # 4) backpropagation: her ağırlığın gradyanı
        optimizer.step()                 # 5) ağırlıkları gradyanın tersi yönünde güncelle


@torch.no_grad()
def evaluate(model, x, y, loss_fn, batch_size=10_000):
    """Verilen veri üzerinde ortalama loss ve accuracy (eval modunda, gradyan hesaplamadan)."""
    model.eval()  # dropout kapalı, batchnorm sabit istatistiklerle
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        total_loss += loss_fn(logits, y[i:i + batch_size]).item() * len(logits)
        correct += (logits.argmax(dim=1) == y[i:i + batch_size]).sum().item()
    return total_loss / len(x), correct / len(x)


def train(config, data, device, verbose=True):
    """config'teki ayarlarla bir model eğitir.

    Her epoch sonunda train ve validation setinin tamamında loss/accuracy ölçülür.
    Eğitim bitince model, validation loss'un en düşük olduğu epoch'un ağırlıklarına döndürülür.
    Süre yalnızca eğitim adımlarını kapsar (ölçüm kısmı hariç).
    """
    cfg = {**DEFAULT_CONFIG, **config}
    warm_up(device)  # seed'den önce: ısınmanın rastgele sayıları eğitimi etkilemesin
    torch.manual_seed(cfg["seed"])  # başlangıç ağırlıkları tekrarlanabilir olsun
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])  # batch sırası da

    model = MLP(cfg["hidden"], cfg["activation"], cfg["dropout"], cfg["batchnorm"]).to(device)
    loss_fn = make_loss_fn(cfg["loss"])
    optimizer = make_optimizer(cfg, model.parameters())
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    history = []
    best = {"val_loss": math.inf, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    epochs_without_improvement = 0
    for epoch in range(1, cfg["epochs"] + 1):
        sync(device)
        start = time.perf_counter()
        train_one_epoch(model, x_train, y_train, loss_fn, optimizer, cfg["batch_size"], generator)
        sync(device)
        epoch_seconds = time.perf_counter() - start

        train_loss, train_acc = evaluate(model, x_train, y_train, loss_fn)
        val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": epoch_seconds})
        if verbose:
            print(f"epoch {epoch:>2}/{cfg['epochs']} | train loss {train_loss:.4f} acc %{train_acc * 100:.2f} | "
                  f"val loss {val_loss:.4f} acc %{val_acc * 100:.2f} | {epoch_seconds:.2f} sn")

        if not math.isfinite(train_loss):
            if verbose:
                print("Loss sonsuz/NaN oldu (eğitim ıraksadı), eğitim durduruldu.")
            break
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg["patience"] is not None and epochs_without_improvement >= cfg["patience"]:
                if verbose:
                    print(f"Early stopping: val loss {cfg['patience']} epoch boyunca iyileşmedi.")
                break

    model.load_state_dict(best["state"])  # en iyi epoch'un ağırlıklarına dön
    return {
        "config": cfg,
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "train_seconds": sum(h["epoch_seconds"] for h in history),
        "n_params": count_parameters(model),
    }
