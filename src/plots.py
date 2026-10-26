"""Eğitim sonuçlarının grafikleri."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import plot_style


def plot_history(history, best_epoch=None, title=None):
    """Tek bir eğitimin epoch'lara göre train/validation loss ve accuracy eğrileri."""
    return plot_histories([history], [best_epoch] if best_epoch else None, title)


def plot_histories(histories, best_epochs=None, title=None):
    """Bir veya daha fazla eğitimin (ör. farklı seed'ler) loss ve accuracy eğrileri.

    Birden fazla eğitim verilirse çizgi ortalamayı, gölge ± standart sapmayı gösterir.
    Kesikli dikey çizgiler, her eğitimin en düşük val loss'lu epoch'udur.
    """
    epochs = np.array([h["epoch"] for h in histories[0]])
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [(ax_loss, "loss", "Loss", 1), (ax_acc, "acc", "Accuracy (%)", 100)]
    for ax, key, label, scale in panels:
        for i, (split, name) in enumerate([("train", "Train"), ("val", "Validation")]):
            values = np.array([[h[f"{split}_{key}"] * scale for h in history] for history in histories])
            mean = values.mean(axis=0)
            color = plot_style.SERIES[i]
            ax.plot(epochs, mean, color=color, marker="o", markersize=4, label=name)
            if len(histories) > 1:
                std = values.std(axis=0, ddof=1)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
        for best in sorted(set(best_epochs or [])):
            ax.axvline(best, color=plot_style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("Epoch")
        ax.set_title(label, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if best_epochs:
        # Çizgi sağ yarıdaysa yazıyı soluna koy ki grafiğin dışına taşmasın
        anchor = max(best_epochs)
        right_half = anchor > (epochs[0] + epochs[-1]) / 2
        text = "en iyi epoch\n(en düşük val loss)" if len(set(best_epochs)) == 1 else "seed'lerin en iyi\nepoch'ları"
        ax_loss.annotate(text, xy=(anchor if not right_half else min(best_epochs), 1.0),
                         xycoords=("data", "axes fraction"), xytext=(-4 if right_half else 4, -4),
                         textcoords="offset points", color=plot_style.MUTED, fontsize=8,
                         ha="right" if right_half else "left", va="top")
    ax_acc.legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout()
    return fig
