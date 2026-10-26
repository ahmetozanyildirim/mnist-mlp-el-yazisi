"""Deney 1 – Temel MLP modeli (baseline).

Baseline ayarlarıyla (train.DEFAULT_CONFIG) modeli birkaç seed ile eğitir. Her seed için en iyi
epoch'taki (en düşük val loss) train/validation metriklerini, test sonucunu ve eğitim süresini
kaydeder. Test seti yalnızca raporlama içindir; hiçbir karar test sonucuna bakarak verilmez.

Çalıştırma:  python src/deney1.py
Çıktılar  :  results/deney1/sonuclar.json, egriler.png, tablo.md
"""
import json
import platform
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import plot_style
from data import load_data
from device import get_device
from plots import plot_histories
from train import DEFAULT_CONFIG, evaluate, make_loss_fn, train

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "deney1"
SEEDS = (0, 1, 2)
METRICS = ["best_epoch", "train_loss", "val_loss", "train_acc", "val_acc", "test_loss", "test_acc", "train_seconds"]


def run(data, device, seeds=SEEDS, verbose=True):
    """Her seed için baseline'ı eğitir, en iyi epoch'un modelini test setinde değerlendirir."""
    loss_fn = make_loss_fn(DEFAULT_CONFIG["loss"])
    x_test, y_test = data["test"]
    runs = []
    for seed in seeds:
        if verbose:
            print(f"\n=== seed {seed} ===")
        result = train({"seed": seed}, data, device, verbose=verbose)
        best = result["history"][result["best_epoch"] - 1]
        # result["model"] en iyi epoch'un ağırlıklarına döndürülmüş durumda
        test_loss, test_acc = evaluate(result["model"], x_test, y_test, loss_fn)
        runs.append({
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "train_loss": best["train_loss"],
            "val_loss": best["val_loss"],
            "train_acc": best["train_acc"],
            "val_acc": best["val_acc"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_seconds": result["train_seconds"],
            "history": result["history"],
        })
    return runs


def summarize(runs):
    """Her metriğin seed'ler üzerinden ortalaması ve standart sapması."""
    summary = {}
    for m in METRICS:
        values = [r[m] for r in runs]
        summary[m] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    return summary


def format_metric(metric, value, std=None):
    """Bir metriği tabloya uygun biçimde yazar; std verilirse 'ortalama ± std' olarak."""
    if metric.endswith("_acc"):
        return f"%{value * 100:.2f}" + (f" ± {std * 100:.2f}" if std is not None else "")
    if metric == "train_seconds":
        return f"{value:.1f}" + (f" ± {std:.1f}" if std is not None else "") + " sn"
    if metric == "best_epoch":
        return f"{value:.1f} ± {std:.1f}" if std is not None else str(value)
    return f"{value:.4f}" + (f" ± {std:.4f}" if std is not None else "")


def markdown_table(runs, summary):
    """Rapora konacak sonuç tablosu (her seed'in en iyi epoch'undaki değerler)."""
    header = ("| Seed | En iyi epoch | Train loss | Val loss | Train acc | Val acc | Test loss | Test acc "
              f"| Eğitim süresi ({DEFAULT_CONFIG['epochs']} epoch) |")
    lines = [header, "|" + "---|" * (len(METRICS) + 1)]
    for r in runs:
        lines.append(f"| {r['seed']} | " + " | ".join(format_metric(m, r[m]) for m in METRICS) + " |")
    means = [format_metric(m, summary[m]["mean"], summary[m]["std"]) for m in METRICS]
    lines.append("| **Ortalama ± std** | " + " | ".join(means) + " |")
    return "\n".join(lines)


def hardware_info(device):
    info = {"pytorch": torch.__version__, "python": platform.python_version(), "cihaz": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    info["cpu"] = platform.processor()
    return info


def save(runs, summary, device, out_dir=RESULTS_DIR):
    """Sonuçları JSON, grafik ve markdown tablo olarak kaydeder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in DEFAULT_CONFIG.items()}
    payload = {"config": config, "seeds": [r["seed"] for r in runs], "donanim": hardware_info(device),
               "ozet": summary, "calistirmalar": runs}
    (out_dir / "sonuclar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "tablo.md").write_text(markdown_table(runs, summary) + "\n", encoding="utf-8")

    fig = plot_histories([r["history"] for r in runs], [r["best_epoch"] for r in runs],
                         title=f"Deney 1 – Baseline MLP: {len(runs)} seed ortalaması ± standart sapma")
    fig.savefig(out_dir / "egriler.png")
    plt.close(fig)
    return out_dir


def main(out_dir=RESULTS_DIR):
    plot_style.apply()
    device = get_device()
    data, _ = load_data(include_test=True, device=device)  # Deney 1 test sonucunu raporlamak için
    runs = run(data, device)
    summary = summarize(runs)
    print("\n" + markdown_table(runs, summary))
    print("\nKaydedildi:", save(runs, summary, device, out_dir))


if __name__ == "__main__":
    main()
