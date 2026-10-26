"""Aşama 1: veri setini tanıma.

Ayrımın boyutlarını, normalizasyon istatistiklerini ve sınıf dağılımını yazdırır;
örnek görselleri ve sınıf dağılımı grafiğini results/ klasörüne kaydeder.

Çalıştırma:  python src/explore_data.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import plot_style
from data import compute_mean_std, get_splits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}


def print_sizes_and_stats(splits):
    print("== Parça boyutları ==")
    for name, (x, _) in splits.items():
        print(f"{SPLIT_NAMES[name]:<11}: {len(x):>6} görüntü, her biri {tuple(x.shape[1:])}, "
              f"piksel aralığı [{x.min().item()}, {x.max().item()}]")

    x_train = splits["train"][0]
    zero_ratio = (x_train == 0).float().mean().item()
    print(f"\nTrain piksellerinin %{zero_ratio * 100:.1f}'i tamamen siyah (0).")

    # Normalizasyon istatistikleri yalnızca train'den hesaplanır.
    # Karşılaştırma için bütün veriden (train+val+test) hesaplanmış hâlini de gösteriyoruz.
    mean, std = compute_mean_std(x_train)
    all_x = torch.cat([x for x, _ in splits.values()])
    leak_mean, leak_std = compute_mean_std(all_x)
    print("\n== Normalizasyon istatistikleri ([0,1] ölçeğinde) ==")
    print(f"Yalnızca train      : ortalama={mean:.4f}, std={std:.4f}   <- kullanılan")
    print(f"Bütün veri (sızıntı): ortalama={leak_mean:.4f}, std={leak_std:.4f}")


def class_percentages(splits):
    return {
        name: torch.bincount(y, minlength=10).float() / len(y) * 100
        for name, (_, y) in splits.items()
    }


def print_class_table(splits, percents):
    print("\n== Sınıf dağılımı (markdown tablo) ==")
    header = "| Rakam | " + " | ".join(SPLIT_NAMES[n] for n in splits) + " |"
    print(header)
    print("|---" * (len(splits) + 1) + "|")
    for digit in range(10):
        cells = []
        for name, (_, y) in splits.items():
            count = (y == digit).sum().item()
            cells.append(f"{count} (%{percents[name][digit]:.1f})")
        print(f"| {digit} | " + " | ".join(cells) + " |")


def plot_samples(splits, per_digit=12, gap=2):
    """Her rakamdan train'deki ilk birkaç örneği tek bir mozaikte gösterir."""
    x, y = splits["train"]
    size = 28 + gap
    mosaic = np.zeros((10 * size - gap, per_digit * size - gap))
    for digit in range(10):
        examples = x[y == digit][:per_digit]
        for j, img in enumerate(examples):
            mosaic[digit * size:digit * size + 28, j * size:j * size + 28] = img.numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    # cmap="gray": 0 siyah, 255 beyaz -> MNIST'in gerçek görünümü (siyah zemin, beyaz rakam)
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([d * size + 14 for d in range(10)], labels=[str(d) for d in range(10)])
    ax.set_xticks([])
    ax.tick_params(left=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("MNIST train örnekleri (her satır bir rakam)", loc="left")
    return fig


def plot_class_distribution(percents):
    names = list(percents)
    digits = np.arange(10)
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(digits + offset, percents[name].numpy(), width=width * 0.92,
               color=plot_style.SERIES[i], label=SPLIT_NAMES[name])
    ax.axhline(10, color=plot_style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("tam denge\n(%10)", xy=(1.01, 10), xycoords=("axes fraction", "data"),
                color=plot_style.MUTED, ha="left", va="center", fontsize=9)
    ax.set_xticks(digits)
    ax.set_xlabel("Rakam")
    ax.set_ylabel("Parça içindeki oran (%)")
    ax.set_ylim(0, 12.5)
    ax.set_title("Sınıf dağılımı: üç parçada oranlar neredeyse aynı", loc="left")
    ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    return fig


def save(fig, filename):
    fig.savefig(RESULTS_DIR / filename)
    plt.close(fig)


def main():
    plot_style.apply()
    RESULTS_DIR.mkdir(exist_ok=True)

    # Burada test setinin yalnızca etiket sayılarına bakıyoruz; modelle hiçbir ilişkisi yok
    splits = get_splits(include_test=True)
    percents = class_percentages(splits)

    print_sizes_and_stats(splits)
    print_class_table(splits, percents)
    save(plot_samples(splits), "data_samples.png")
    save(plot_class_distribution(percents), "class_distribution.png")
    print(f"\nGrafikler kaydedildi: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
