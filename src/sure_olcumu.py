"""Bir epoch'luk eğitimin süresini ölçer: CPU ve GPU karşılaştırması için (Aşama 5).

Gerçek veri yerine aynı boyutta rastgele veri kullanılır; süre verinin değerlerine değil boyutuna bağlıdır. Bu sayede
script yalnızca PyTorch'a ihtiyaç duyar ve farklı Python ortamlarında (ör. Intel MKL'li CPU PyTorch'u) aynen
çalıştırılabilir. Ölçüm, train.py'deki eğitim döngüsüyle aynı adımları yapar.

Çalıştırma:  python src/sure_olcumu.py --device cpu --hidden 256 256 --batch-size 64
Çıktı     :  tek satır JSON (ortam bilgisi ve epoch başına saniye)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

from model import MLP

N_TRAIN = 50_000


def time_epochs(device, hidden, batch_size, optimizer_name, lr, epochs=2):
    torch.manual_seed(0)
    x = torch.randn(N_TRAIN, 784, device=device)
    y = torch.randint(0, 10, (N_TRAIN,), device=device)
    model = MLP(tuple(hidden)).to(device)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if optimizer_name == "sgd_momentum" else 0)
    loss_fn = nn.CrossEntropyLoss()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def steps(count=None):
        perm = torch.randperm(N_TRAIN, device=device)
        for n, i in enumerate(range(0, N_TRAIN, batch_size)):
            if count is not None and n == count:
                break
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    steps(50)  # ısınma: ilk adımlardaki kütüphane yükleme ve bellek ayırma maliyeti ölçüme girmesin
    times = []
    for _ in range(epochs):
        sync()
        start = time.perf_counter()
        steps()
        sync()
        times.append(time.perf_counter() - start)
    return times


def environment(device):
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "mkl": torch.backends.mkl.is_available(), "threads": torch.get_num_threads(), "cihaz": str(device)}
    info["islemci"] = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    return info


def measure_in(python, device, hidden, batch_size, optimizer_name="sgd_momentum", lr=0.01, epochs=2):
    """Bu script'i verilen Python yorumlayıcısıyla ayrı bir süreçte çalıştırır ve sonucunu döner."""
    command = [str(python), str(Path(__file__).resolve()), "--device", device, "--hidden", *map(str, hidden),
               "--batch-size", str(batch_size), "--optimizer", optimizer_name, "--lr", str(lr), "--epochs", str(epochs)]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return json.loads(output.strip().splitlines()[-1])


def timing_table(rows):
    """{ad: measure_in sonucu} → markdown tablo; hız, en yavaşa göre."""
    slowest = max(min(r["epoch_saniye"]) for r in rows.values())
    lines = ["| Ortam | PyTorch | MKL | İş parçacığı | İşlemci | Epoch süresi | Hız (en yavaşa göre) |",
             "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        seconds = min(r["epoch_saniye"])
        lines.append(f"| {name} | {r['torch']} | {'var' if r['mkl'] else 'yok'} | {r['threads']} | {r['islemci']} | "
                     f"{seconds:.2f} sn | {slowest / seconds:.1f}× |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", default="sgd_momentum")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.device == "cuda":
        from device import get_device
        device = get_device()
    else:
        device = torch.device("cpu")
    times = time_epochs(device, args.hidden, args.batch_size, args.optimizer, args.lr, args.epochs)
    print(json.dumps({**environment(device), "epoch_saniye": times}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
