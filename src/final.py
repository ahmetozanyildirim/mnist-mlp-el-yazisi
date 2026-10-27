"""Aşama 5: final model ve test setinde TEK seferlik değerlendirme.

Aşama 4C'de seçilen ayar (results/arama/secilen_ayar.json) birkaç seed ile eğitilir. Her seed'in en iyi
epoch'undaki modeli test setinde bir kez değerlendirilir. Test sonucu yalnızca raporlanır; ona bakarak hiçbir
ayar değiştirilmez. Validation accuracy'si en yüksek seed'in modeli, kendi el yazımızı tanımak için (Aşama 6)
results/final/model.pt olarak kaydedilir.

Çalıştırma:  python src/final.py
Çıktılar  :  results/final/sonuclar.json, tablo.md, egriler.png, karisiklik_matrisi.png, hatalar.png, model.pt
"""
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import plot_style
from data import load_data
from deney1 import hardware_info
from device import get_device
from model import MLP
from plots import plot_histories
from train import DEFAULT_CONFIG, evaluate, make_loss_fn, train

ROOT = Path(__file__).resolve().parent.parent
SELECTED_CONFIG = ROOT / "results" / "arama" / "secilen_ayar.json"
RESULTS_DIR = ROOT / "results" / "final"
SEEDS = (0, 1, 2)
METRICS = ["best_epoch", "train_loss", "val_loss", "train_acc", "val_acc", "test_loss", "test_acc", "train_seconds"]


def load_selected_config(path=SELECTED_CONFIG):
    """Aşama 4C'nin seçtiği ayar (yalnızca DEFAULT_CONFIG'ten farklı olan anahtarlar)."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))["ayar"]
    if "hidden" in config:
        config["hidden"] = tuple(config["hidden"])
    return config


def train_final(config, data, device, seeds=SEEDS, verbose=True):
    """Final ayarı her seed ile eğitir; modeller en iyi epoch'un ağırlıklarıyla döner. Test setine dokunmaz."""
    results = []
    for seed in seeds:
        result = train({**config, "seed": seed}, data, device, verbose=False)
        best = result["history"][result["best_epoch"] - 1]
        if verbose:
            print(f"seed {seed}: en iyi epoch {result['best_epoch']:>2} / {len(result['history'])} | "
                  f"val acc %{best['val_acc'] * 100:.2f} | val loss {best['val_loss']:.4f} | "
                  f"{result['train_seconds']:.1f} sn")
        results.append(result)
    return results


def evaluate_on_test(results, test, loss_name):
    """Her modeli test setinde BİR KEZ değerlendirir ve rapor satırlarını döner."""
    loss_fn = make_loss_fn(loss_name)
    runs = []
    for result in results:
        best = result["history"][result["best_epoch"] - 1]
        test_loss, test_acc = evaluate(result["model"], *test, loss_fn)
        runs.append({"seed": result["config"]["seed"], "best_epoch": result["best_epoch"],
                     "train_loss": best["train_loss"], "val_loss": best["val_loss"],
                     "train_acc": best["train_acc"], "val_acc": best["val_acc"],
                     "test_loss": test_loss, "test_acc": test_acc,
                     "train_seconds": result["train_seconds"], "history": result["history"]})
    return runs


def summarize(runs):
    summary = {}
    for m in METRICS:
        values = [r[m] for r in runs]
        summary[m] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    return summary


def format_metric(metric, value, std=None):
    if metric.endswith("_acc"):
        return f"%{value * 100:.2f}" + (f" ± {std * 100:.2f}" if std is not None else "")
    if metric == "train_seconds":
        return f"{value:.1f}" + (f" ± {std:.1f}" if std is not None else "") + " sn"
    if metric == "best_epoch":
        return f"{value:.1f} ± {std:.1f}" if std is not None else str(value)
    return f"{value:.4f}" + (f" ± {std:.4f}" if std is not None else "")


def markdown_table(runs, summary):
    lines = ["| Seed | En iyi epoch | Train loss | Val loss | Train acc | Val acc | Test loss | Test acc | Eğitim süresi |",
             "|" + "---|" * (len(METRICS) + 1)]
    for r in runs:
        lines.append(f"| {r['seed']} | " + " | ".join(format_metric(m, r[m]) for m in METRICS) + " |")
    means = [format_metric(m, summary[m]["mean"], summary[m]["std"]) for m in METRICS]
    lines.append("| **Ortalama ± std** | " + " | ".join(means) + " |")
    return "\n".join(lines)


@torch.no_grad()
def predict(model, x, batch_size=10_000):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).softmax(dim=1) for i in range(0, len(x), batch_size)])


def confusion_matrix(predictions, labels, num_classes=10):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, pred in zip(labels.tolist(), predictions.tolist()):
        matrix[true, pred] += 1
    return matrix


def plot_confusion(matrix, title=None):
    """Karışıklık matrisi: satır gerçek rakam, sütun tahmin. Köşegen dışındaki her hücre bir hata türü."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    errors = matrix.clone().fill_diagonal_(0).numpy()
    ax.imshow(errors, cmap="Oranges")
    for i in range(10):
        for j in range(10):
            value = int(matrix[i, j])
            color = plot_style.INK_SECONDARY if i == j else (plot_style.INK if errors[i, j] else plot_style.MUTED)
            ax.text(j, i, value, ha="center", va="center", fontsize=8, color=color,
                    fontweight="semibold" if i != j and errors[i, j] >= max(3, errors.max() / 3) else "normal")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Tahmin edilen rakam")
    ax.set_ylabel("Gerçek rakam")
    ax.grid(False)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def plot_mistakes(images, labels, predictions, probabilities, mean, std, count=24, title=None):
    """Modelin yanlış bildiği örneklerden en emin olduğu yanlışlar."""
    wrong = (predictions != labels).nonzero().flatten()
    confidence = probabilities[wrong, predictions[wrong]]
    order = wrong[confidence.argsort(descending=True)][:count]
    cols = 8
    rows = math.ceil(len(order) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.55))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, order.tolist()):
        image = (images[i].reshape(28, 28) * std + mean).clamp(0, 1).cpu().numpy()
        ax.imshow(image, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"gerçek {labels[i]}\ntahmin {predictions[i]} (%{probabilities[i, predictions[i]] * 100:.0f})",
                     fontsize=8, color=plot_style.INK_SECONDARY)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout()
    return fig


def save_model(result, normalization_stats, path):
    """Modeli, onu yeniden kurmak için gereken ayar ve normalizasyon istatistikleriyle birlikte kaydeder."""
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in result["config"].items()}
    torch.save({"config": config, "state_dict": result["model"].state_dict(),
                "normalization": {"mean": normalization_stats[0], "std": normalization_stats[1]}}, path)


def load_model(path, device="cpu"):
    """Kaydedilmiş modeli ve normalizasyon istatistiklerini (ortalama, std) yükler."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = {**DEFAULT_CONFIG, **checkpoint["config"]}
    model = MLP(tuple(cfg["hidden"]), cfg["activation"], cfg["dropout"], cfg["batchnorm"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    stats = checkpoint["normalization"]
    return model, (stats["mean"], stats["std"])


def save(config, results, runs, summary, data, stats, device, out_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plain_config = {k: list(v) if isinstance(v, tuple) else v for k, v in {**DEFAULT_CONFIG, **config}.items()}
    # Aşama 6'da kullanılacak model: validation accuracy'si en yüksek seed (seçim yalnızca validation'a göre)
    chosen = max(range(len(runs)), key=lambda i: runs[i]["val_acc"])
    payload = {"ayar": plain_config, "seeds": [r["seed"] for r in runs], "donanim": hardware_info(device),
               "ozet": summary, "kaydedilen_model_seed": runs[chosen]["seed"], "calistirmalar": runs}
    (out_dir / "sonuclar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "tablo.md").write_text(markdown_table(runs, summary) + "\n", encoding="utf-8")
    save_model(results[chosen], stats, out_dir / "model.pt")

    fig = plot_histories([r["history"] for r in runs], [r["best_epoch"] for r in runs],
                         title=f"Final model: {len(runs)} seed ortalaması ± standart sapma")
    fig.savefig(out_dir / "egriler.png")
    plt.close(fig)

    x_test, y_test = data["test"]
    probabilities = predict(results[chosen]["model"], x_test)
    predictions = probabilities.argmax(dim=1)
    fig = plot_confusion(confusion_matrix(predictions.cpu(), y_test.cpu()),
                         title=f"Test seti karışıklık matrisi (seed {runs[chosen]['seed']})")
    fig.savefig(out_dir / "karisiklik_matrisi.png")
    plt.close(fig)
    fig = plot_mistakes(x_test, y_test, predictions, probabilities, *stats,
                        title="Modelin en emin olduğu yanlışlar (test seti)")
    fig.savefig(out_dir / "hatalar.png")
    plt.close(fig)
    return out_dir


def main(out_dir=RESULTS_DIR):
    plot_style.apply()
    device = get_device()
    config = load_selected_config()
    print("Final ayar:", config)
    data, stats = load_data(include_test=True, device=device)  # test yalnızca en sondaki değerlendirme için
    results = train_final(config, data, device)
    runs = evaluate_on_test(results, data["test"], {**DEFAULT_CONFIG, **config}["loss"])
    summary = summarize(runs)
    print("\n" + markdown_table(runs, summary))
    print("\nKaydedildi:", save(config, results, runs, summary, data, stats, device, out_dir))


if __name__ == "__main__":
    main()
