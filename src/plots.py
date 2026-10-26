"""Eğitim sonuçlarının grafikleri."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

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


def _stack(histories, key, scale):
    """Farklı uzunluktaki (ör. early stopping) eğrileri NaN ile doldurup tek dizide toplar."""
    length = max(len(h) for h in histories)
    values = np.full((len(histories), length), np.nan)
    for i, history in enumerate(histories):
        values[i, :len(history)] = [e[key] * scale for e in history]
    return np.arange(1, length + 1), values


DEFAULT_COMPARISON_METRICS = (("val_loss", "Validation loss", 1), ("val_acc", "Validation accuracy (%)", 100))


def plot_comparison(results, title=None, metrics=None):
    """Bir karşılaştırma deneyindeki seçeneklerin eğrileri (varsayılan: validation loss ve accuracy).

    metrics: iki panelde çizilecek (anahtar, başlık, çarpan) üçlüleri. Seçeneklerin loss'ları
    farklı ölçeklerdeyse (ör. farklı loss fonksiyonları) loss yerine accuracy çizilmelidir.
    Her seçenek bir renk; çizgi seed'lerin ortalaması, gölge ± standart sapma.
    Iraksayan veya hiç öğrenemeyen eğitimler ölçeği bozmasın diye çizilmez, açıklamada belirtilir.
    """
    metrics = metrics or DEFAULT_COMPARISON_METRICS
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3.8))
    for i, res in enumerate(results):
        color = plot_style.SERIES[i]
        histories = [r["history"] for r in res["runs"] if not r["failed"]]
        if not histories:
            reason = "ıraksadı" if any(r["diverged"] for r in res["runs"]) else "öğrenemedi, çizilmedi"
            axes[-1].plot([], [], color=color, label=f"{res['label']} ({reason})")
            continue
        for ax, (key, _, scale) in zip(axes, metrics):
            epochs, values = _stack(histories, key, scale)
            mean = np.nanmean(values, axis=0)
            ax.plot(epochs, mean, color=color, marker="o", markersize=3,
                    label=res["label"] if ax is axes[-1] else None)
            if len(histories) > 1:
                std = np.nanstd(values, axis=0, ddof=1)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    for ax, (_, label, _) in zip(axes, metrics):
        ax.set_xlabel("Epoch")
        ax.set_title(label, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout()
    return fig


def plot_lr_sweep(results, group_key, group_names, title=None):
    """Learning rate taraması: her grup (ör. her optimizer) için learning rate'e göre validation sonuçları.

    Her nokta bir ayarın seed ortalaması, hata çubuğu ± standart sapma (en düşük val loss'lu epoch'ta).
    Iraksayan veya hiç öğrenemeyen ayarlar ölçeği bozmasın diye grafiğin altına × ile işaretlenir.
    """
    metrics = (("val_acc", "Validation accuracy (%)", 100), ("val_loss", "Validation loss (en düşük)", 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3.8))
    groups = list(dict.fromkeys(res["config"][group_key] for res in results))
    any_failed = False
    for i, group in enumerate(groups):
        color = plot_style.SERIES[i]
        members = sorted((res for res in results if res["config"][group_key] == group),
                         key=lambda res: res["config"]["lr"])
        for ax, (key, _, scale) in zip(axes, metrics):
            lrs, means, stds, failed = [], [], [], []
            for res in members:
                values = [r[key] * scale for r in res["runs"] if not r["failed"]]
                if not values:
                    failed.append(res["config"]["lr"])
                    continue
                lrs.append(res["config"]["lr"])
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            label = group_names.get(group, group) if ax is axes[0] else None
            if lrs:
                ax.errorbar(lrs, means, yerr=stds, color=color, marker="o", markersize=4, capsize=3, label=label)
            else:
                ax.plot([], [], color=color, marker="o", label=label)
            if failed:
                any_failed = True
                ax.plot(failed, [0.04] * len(failed), linestyle="none", marker="x", markersize=8,
                        markeredgewidth=2, color=color, transform=ax.get_xaxis_transform(), clip_on=False)

    for ax, (_, label, _) in zip(axes, metrics):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: np.format_float_positional(v, trim="-")))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Learning rate (logaritmik eksen)")
        ax.set_title(label, loc="left")
    # Açıklama veriyi kapatmasın diye grafiklerin altında
    handles, labels = axes[0].get_legend_handles_labels()
    if any_failed:
        handles.append(plt.Line2D([], [], linestyle="none", marker="x", markeredgewidth=2, color=plot_style.MUTED))
        labels.append("öğrenemedi / ıraksadı")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig
