"""Aşama 4A-8: epoch sayısı, early stopping ve model seçme ölçütü.

Uzun bir eğitimin (100 epoch) her epoch'taki ölçümleri kaydedilir. Farklı kurallar ("en fazla 20 epoch eğit, en
düşük val loss'lu epoch'u seç", "val loss 5 epoch iyileşmezse dur" ...) bu geçmiş üzerinde sonradan uygulanır.
Bu, her kural için ayrı eğitim yapmakla aynı sonucu verir: early stopping eğitimi yalnızca erken keser, kesilene
kadarki epoch'lar birebir aynıdır.

Validation seti ikiye bölünmüştür (karsilastirma.split_holdout):
- "val" (5 000 görüntü): kurallar epoch'u buna bakarak seçer.
- "holdout" (5 000 görüntü): seçilen epoch'un gerçekte ne kadar iyi olduğu buna bakarak ölçülür.
Seçimi yapan veriyle seçimin sonucunu ölçmek iyimser bir sonuç verir (seçim yanlılığı); holdout bunu önler.
"""
import math
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import plot_style

# ad: (seçim ölçütü, en fazla epoch, patience). patience None ise early stopping yok.
POLICIES = {
    "10 epoch, en düşük val loss": ("val_loss", 10, None),
    "20 epoch, en düşük val loss (mevcut)": ("val_loss", 20, None),
    "40 epoch, en düşük val loss": ("val_loss", 40, None),
    "100 epoch, en düşük val loss": ("val_loss", 100, None),
    "20 epoch, en yüksek val acc": ("val_acc", 20, None),
    "100 epoch, en yüksek val acc": ("val_acc", 100, None),
    "Early stopping: val loss, patience 5": ("val_loss", 100, 5),
    "Early stopping: val loss, patience 10": ("val_loss", 100, 10),
    "Early stopping: val acc, patience 5": ("val_acc", 100, 5),
    "Early stopping: val acc, patience 10": ("val_acc", 100, 10),
}
CRITERION_NAMES = {"val_loss": "en düşük val loss", "val_acc": "en yüksek val acc"}


def apply_policy(history, criterion, max_epochs, patience=None):
    """Bir kuralı bir eğitim geçmişine uygular: eğitim hangi epoch'ta durur, hangi epoch'un modeli seçilir?

    train() içindeki early stopping ile aynı mantık: ölçüt iyileşmezse sayaç artar, patience'a ulaşınca durulur.
    """
    best_score, best_epoch, waited = math.inf, 0, 0
    stop = min(max_epochs, len(history))
    for h in history[:stop]:
        score = h["val_loss"] if criterion == "val_loss" else -h["val_acc"]
        if score < best_score:
            best_score, best_epoch, waited = score, h["epoch"], 0
        else:
            waited += 1
            if patience is not None and waited >= patience:
                stop = h["epoch"]
                break
    return {"stop_epoch": stop, "chosen_epoch": best_epoch,
            "seconds": sum(h["epoch_seconds"] for h in history[:stop])}


def _mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def evaluate_policies(results, policies=POLICIES):
    """Her kural için seed'lerin ortalaması ± std: durma epoch'u, seçilen epoch, val (seçim yarısı) ve holdout sonuçları."""
    runs = [run for res in results for run in res["runs"]]
    rows = []
    for name, (criterion, max_epochs, patience) in policies.items():
        per_run = []
        for run in runs:
            outcome = apply_policy(run["history"], criterion, max_epochs, patience)
            chosen = run["history"][outcome["chosen_epoch"] - 1]
            per_run.append({**outcome, "val_acc": chosen["val_acc"], "holdout_acc": chosen["holdout_acc"],
                            "holdout_loss": chosen["holdout_loss"]})
        row = {"name": name, "criterion": criterion, "max_epochs": max_epochs, "patience": patience}
        for key in ("stop_epoch", "chosen_epoch", "seconds", "val_acc", "holdout_acc", "holdout_loss"):
            row[key] = _mean_std([r[key] for r in per_run])
        rows.append(row)
    return rows


def choose_policy(rows, tolerance):
    """Kural seçimi (sonuçlardan önce belirlendi).

    1. Birincil ölçüt holdout accuracy. En iyisine eşikten (0.3 puan) yakın kurallar eşit derecede iyi sayılır.
    2. Bu kurallar 4B'de çok farklı ayarlarla da kullanılacağı için, ayara kendiliğinden uyum sağlayan early
       stopping kuralları tercih edilir; sabit epoch sayısı yalnızca bu ayar için doğru olabilir.
    3. Eşit derecede iyi early stopping kurallarından en kısa sürede biten seçilir.
    Hiçbir early stopping kuralı eşiğin içinde değilse, eşik içindeki en hızlı kural seçilir.
    """
    top = max(row["holdout_acc"][0] for row in rows)
    good = [row for row in rows if top - row["holdout_acc"][0] < tolerance]
    adaptive = [row for row in good if row["patience"] is not None]
    return min(adaptive or good, key=lambda row: row["seconds"][0])


def policy_table(rows):
    lines = ["| Kural | Durduğu epoch | Seçilen epoch | Val acc (seçim yarısı) | Holdout acc | Holdout loss | "
             "Eğitim süresi |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        (stop, stop_sd), (chosen, chosen_sd) = row["stop_epoch"], row["chosen_epoch"]
        (val, val_sd), (hold, hold_sd), (loss, loss_sd) = row["val_acc"], row["holdout_acc"], row["holdout_loss"]
        secs, secs_sd = row["seconds"]
        lines.append(f"| {row['name']} | {stop:.1f} ± {stop_sd:.1f} | {chosen:.1f} ± {chosen_sd:.1f} | "
                     f"%{val * 100:.2f} ± {val_sd * 100:.2f} | %{hold * 100:.2f} ± {hold_sd * 100:.2f} | "
                     f"{loss:.4f} ± {loss_sd:.4f} | {secs:.1f} ± {secs_sd:.1f} sn |")
    return "\n".join(lines)


def policy_ranking(rows, tolerance):
    top = max(row["holdout_acc"][0] for row in rows)
    chosen = choose_policy(rows, tolerance)
    lines = []
    for row in sorted(rows, key=lambda row: -row["holdout_acc"][0]):
        gap = top - row["holdout_acc"][0]
        verdict = "en iyi" if gap == 0 else ("fark anlamlı DEĞİL" if gap < tolerance else "anlamlı şekilde kötü")
        note = "  <- SEÇİLEN" if row is chosen else ""
        lines.append(f"{row['name']:<40} holdout acc %{row['holdout_acc'][0] * 100:.2f} | en iyiden fark "
                     f"{gap * 100:.2f} puan -> {verdict:<20} | {row['seconds'][0]:5.1f} sn{note}")
    return "\n".join(lines)


def policy_report(results, tolerance):
    rows = evaluate_policies(results)
    return policy_table(rows) + "\n\n" + policy_ranking(rows, tolerance)


def plot_long_training(results, title=None, marks=(10, 20, 40)):
    """Uzun eğitimin eğrileri: train, val (seçim yarısı) ve holdout; seed ortalaması ± std."""
    runs = [run for res in results for run in res["runs"]]
    epochs = np.array([h["epoch"] for h in runs[0]["history"]])
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    series = [("train", "Train"), ("val", "Val (seçim yarısı)"), ("holdout", "Holdout")]
    for ax, key, label, scale in [(ax_loss, "loss", "Loss", 1), (ax_acc, "acc", "Accuracy (%)", 100)]:
        for i, (split, name) in enumerate(series):
            values = np.array([[h[f"{split}_{key}"] * scale for h in run["history"]] for run in runs])
            mean = values.mean(axis=0)
            ax.plot(epochs, mean, color=plot_style.SERIES[i], linewidth=1.6, label=name)
            if len(runs) > 1:
                std = values.std(axis=0, ddof=1)
                ax.fill_between(epochs, mean - std, mean + std, color=plot_style.SERIES[i], alpha=0.15, linewidth=0)
        for mark in marks:
            ax.axvline(mark, color=plot_style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("Epoch")
        ax.set_title(label, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_acc.set_ylim(97, 100.2)
    ax_acc.legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout()
    return fig


def save_policy_analysis(results, out_dir, tolerance):
    rows = evaluate_policies(results)
    (out_dir / "kurallar.md").write_text(policy_table(rows) + "\n\n```\n" + policy_ranking(rows, tolerance)
                                         + "\n```\n", encoding="utf-8")
    fig = plot_long_training(results, title=f"{len(results[0]['runs'][0]['history'])} epoch'luk eğitim: "
                                            f"{len(results[0]['runs'])} seed ortalaması ± std")
    fig.savefig(out_dir / "uzun_egitim.png")
    plt.close(fig)
