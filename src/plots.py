"""Eğitim sonuçlarının grafikleri."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import plot_style


def plot_history(history, best_epoch=None, title=None):
    """Epoch'lara göre train/validation loss ve accuracy eğrileri (yan yana iki grafik)."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [(ax_loss, "loss", "Loss", 1), (ax_acc, "acc", "Accuracy (%)", 100)]
    for ax, key, label, scale in panels:
        ax.plot(epochs, [h[f"train_{key}"] * scale for h in history],
                color=plot_style.SERIES[0], marker="o", markersize=4, label="Train")
        ax.plot(epochs, [h[f"val_{key}"] * scale for h in history],
                color=plot_style.SERIES[1], marker="o", markersize=4, label="Validation")
        if best_epoch:
            ax.axvline(best_epoch, color=plot_style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("Epoch")
        ax.set_title(label, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if best_epoch:
        # Çizgi sağ yarıdaysa yazıyı soluna koy ki grafiğin dışına taşmasın
        right_half = best_epoch > (epochs[0] + epochs[-1]) / 2
        ax_loss.annotate("en iyi epoch\n(en düşük val loss)", xy=(best_epoch, 1.0),
                         xycoords=("data", "axes fraction"), xytext=(-4 if right_half else 4, -4),
                         textcoords="offset points", color=plot_style.MUTED, fontsize=8,
                         ha="right" if right_half else "left", va="top")
    ax_acc.legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout()
    return fig
