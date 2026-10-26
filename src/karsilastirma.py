"""Aşama 4A: hiperparametreleri tek tek karşılaştırma deneyleri.

Her deney, bir başlangıç ayarından yalnızca TEK bir hiperparametreyi değiştiren seçenekleri
birkaç seed ile eğitir ve validation sonuçlarını karşılaştırır. Bu deneylerde test seti hiç
yüklenmez; bütün kararlar validation'a göre verilir.

Çalıştırma:  python src/karsilastirma.py <deney_adı>      (ör. python src/karsilastirma.py 4a1_normalizasyon)
Çıktılar  :  results/karsilastirma/<deney_adı>/sonuclar.json, tablo.md, egriler.png
"""
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style
from data import load_data
from device import get_device
from plots import plot_comparison, plot_lr_sweep
from train import DEFAULT_CONFIG, train

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "karsilastirma"
SEEDS = (0, 1, 2)

# İki seçeneğin val accuracy ortalamaları arasındaki fark bundan küçükse "anlamlı fark yok" sayılır.
# Deney 1'de seed'ler arası std ~0.2 puan; 3 seed'in ortalamalarını karşılaştırırken ~0.3 puanlık
# bir fark seed gürültüsünden ayırt edilemez.
TOLERANCE = 0.003

# Her deneyin başlangıç ayarı = bir önceki deneyin başlangıcı + o deneyde verilen karar.
# Böylece her deney, hangi kararlardan sonra yapıldıysa birebir o hâliyle yeniden çalıştırılabilir.
BASE_4A1 = {}  # Deney 1'in ayarları (train.DEFAULT_CONFIG)
# Karar 4A-1 (README 5.1): standardizasyon. Baseline'daki ayar korundu.
BASE_4A2 = {**BASE_4A1, "normalization": "standard"}
# Karar 4A-2 (README 5.2): cross-entropy. Baseline'daki ayar korundu.
BASE_4A3 = {**BASE_4A2, "loss": "cross_entropy"}
# Karar 4A-3 (README 5.3): momentumlu SGD, lr = 0.01 (baseline: Adam, lr = 0.001).
BASE_4A4 = {**BASE_4A3, "optimizer": "sgd_momentum", "lr": 0.01}

# 4A-3: her optimizer kendine uygun bir learning rate aralığında denenir. Değerler ~3 kat aralıklı.
# Momentumlu SGD'nin aralığı düz SGD'ninkinin 1/10'u: momentum 0.9 iken etkin adım ~lr / (1 − 0.9) = 10 × lr.
# Adam'ın aralığı bir değer daha aşağı uzatıldı: ön denemede en iyi Adam değeri aralığın en küçüğü (0.0001) çıktı.
OPTIMIZER_NAMES = {"sgd": "SGD", "sgd_momentum": "Momentumlu SGD", "adam": "Adam"}
LR_GRID = {
    "sgd": (0.01, 0.03, 0.1, 0.3),
    "sgd_momentum": (0.001, 0.003, 0.01, 0.03),
    "adam": (0.00003, 0.0001, 0.0003, 0.001, 0.003),
}


def format_lr(lr):
    """Learning rate'i bilimsel gösterim (3e-05) yerine ondalık sayı (0.00003) olarak yazar."""
    return np.format_float_positional(lr, trim="-")


EXPERIMENTS = {
    "4a1_normalizasyon": {
        "title": "Girdi normalizasyonu (Adam)",
        "base": BASE_4A1,
        "variants": {
            "Ham piksel (0–255)": {"normalization": "none"},
            "Min-max (0–1)": {"normalization": "minmax"},
            "Standardizasyon": {"normalization": "standard"},
        },
    },
    # Ek analiz (karar için değil): optimizer Adam yerine düz SGD olsaydı normalizasyon ne kadar önemli olurdu?
    "4a1b_normalizasyon_sgd": {
        "title": "Girdi normalizasyonu (ek analiz: düz SGD, lr = 0.1)",
        "base": {**BASE_4A1, "optimizer": "sgd", "lr": 0.1},
        "variants": {
            "Ham piksel (0–255)": {"normalization": "none"},
            "Min-max (0–1)": {"normalization": "minmax"},
            "Standardizasyon": {"normalization": "standard"},
        },
    },
    "4a2_loss": {
        "title": "Loss fonksiyonu",
        "base": BASE_4A2,
        "variants": {
            "Cross-entropy": {"loss": "cross_entropy"},
            "MSE (softmax + one-hot)": {"loss": "mse"},
            "Hinge (multi-margin)": {"loss": "multi_margin"},
        },
        # Loss değerleri farklı ölçeklerde, birbiriyle kıyaslanamaz; grafikte accuracy çizilir
        "plot_metrics": (("train_acc", "Train accuracy (%)", 100), ("val_acc", "Validation accuracy (%)", 100)),
    },
    # Ek analiz (karar için değil): Adam küçük gradyanları telafi ediyor olabilir; düz SGD'de fark açılıyor mu?
    "4a2b_loss_sgd": {
        "title": "Loss fonksiyonu (ek analiz: düz SGD, lr = 0.1)",
        "base": {**BASE_4A2, "optimizer": "sgd", "lr": 0.1},
        "variants": {
            "Cross-entropy": {"loss": "cross_entropy"},
            "MSE (softmax + one-hot)": {"loss": "mse"},
            "Hinge (multi-margin)": {"loss": "multi_margin"},
        },
        "plot_metrics": (("train_acc", "Train accuracy (%)", 100), ("val_acc", "Validation accuracy (%)", 100)),
    },
    "4a3_optimizer_lr": {
        "title": "Optimizer ve learning rate",
        "base": BASE_4A3,
        "variants": {f"{OPTIMIZER_NAMES[opt]}, lr = {format_lr(lr)}": {"optimizer": opt, "lr": lr}
                     for opt, lrs in LR_GRID.items() for lr in lrs},
        # Optimizer'lar, her birinin kendi en iyi learning rate'iyle karşılaştırılır
        "group_by": "optimizer",
        "group_names": OPTIMIZER_NAMES,
    },
}


def summarize_run(result):
    """Bir eğitimin karşılaştırmada kullanılacak özet değerleri (en düşük val loss'lu epoch'ta)."""
    history = result["history"]
    diverged = not math.isfinite(history[-1]["train_loss"])
    best = history[result["best_epoch"] - 1] if result["best_epoch"] > 0 else history[-1]
    max_val_acc = max(h["val_acc"] for h in history)
    return {
        "seed": result["config"]["seed"],
        "diverged": diverged,
        # Loss NaN olmasa da öğrenmeyen eğitim: hiçbir epoch'ta şans seviyesini (%10) belirgin geçemedi
        "failed": diverged or max_val_acc < 0.2,
        "best_epoch": result["best_epoch"],
        "val_acc": best["val_acc"],
        "val_loss": best["val_loss"],
        "train_acc": best["train_acc"],
        "first_val_acc": history[0]["val_acc"],
        "max_val_acc": max_val_acc,
        "train_seconds": result["train_seconds"],
        "history": history,
    }


def failure_text(runs):
    if any(r["diverged"] for r in runs):
        return "ıraksadı (loss NaN)"
    return f"öğrenemedi (hep aynı tahmin, val acc %{runs[0]['val_acc'] * 100:.2f})"


def run_experiment(name, device, seeds=SEEDS, verbose=True, only=None):
    """Bir deneyin bütün seçeneklerini her seed ile eğitir.

    only: yalnızca bu ayarlara sahip seçenekleri eğitir (ör. {"optimizer": "sgd"}).
    """
    experiment = EXPERIMENTS[name]
    data_cache = {}
    results = []
    for label, change in experiment["variants"].items():
        config = {**experiment["base"], **change}
        if only and any(config.get(key) != value for key, value in only.items()):
            continue
        normalization = config.get("normalization", DEFAULT_CONFIG["normalization"])
        if normalization not in data_cache:
            data_cache[normalization], _ = load_data(normalization=normalization, device=device)  # test yüklenmez
        runs = []
        for seed in seeds:
            run = summarize_run(train({**config, "seed": seed}, data_cache[normalization], device, verbose=False))
            runs.append(run)
            if verbose:
                status = failure_text([run]) if run["failed"] else (
                    f"en iyi epoch {run['best_epoch']:>2} | val acc %{run['val_acc'] * 100:.2f} | "
                    f"val loss {run['val_loss']:.4f}")
                print(f"{label:<28} seed {seed}: {status} | {run['train_seconds']:.1f} sn")
        results.append({"label": label, "config": config, "runs": runs})
    return results


def mean_std(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def markdown_table(results, loss_comparable=True):
    """Seçeneklerin karşılaştırma tablosu (3 seed ortalaması ± std, en düşük val loss'lu epoch'ta).

    loss_comparable=False: seçenekler farklı loss fonksiyonları kullanıyorsa loss sütunu kıyaslanamaz.
    """
    epochs = DEFAULT_CONFIG["epochs"]
    loss_header = "Val loss (en düşük)" if loss_comparable else "Val loss (her biri kendi loss'u, kıyaslanamaz)"
    lines = [f"| Seçenek | Val acc | {loss_header} | 1. epoch val acc | Train acc | En iyi epoch | "
             f"Eğitim süresi ({epochs} epoch) |",
             "|---|---|---|---|---|---|---|"]
    for res in results:
        runs = [r for r in res["runs"] if not r["failed"]]
        if not runs:
            lines.append(f"| {res['label']} | {failure_text(res['runs'])} | – | – | – | – | – |")
            continue
        label = res["label"]
        if len(runs) < len(res["runs"]):
            label += f" ({len(res['runs']) - len(runs)}/{len(res['runs'])} seed öğrenemedi)"
        acc, acc_sd = mean_std([r["val_acc"] for r in runs])
        loss, loss_sd = mean_std([r["val_loss"] for r in runs])
        first, first_sd = mean_std([r["first_val_acc"] for r in runs])
        train_acc, train_sd = mean_std([r["train_acc"] for r in runs])
        best, best_sd = mean_std([r["best_epoch"] for r in runs])
        secs, secs_sd = mean_std([r["train_seconds"] for r in runs])
        lines.append(f"| {label} | %{acc * 100:.2f} ± {acc_sd * 100:.2f} | {loss:.4f} ± {loss_sd:.4f} | "
                     f"%{first * 100:.2f} ± {first_sd * 100:.2f} | %{train_acc * 100:.2f} ± {train_sd * 100:.2f} | "
                     f"{best:.1f} ± {best_sd:.1f} | {secs:.1f} ± {secs_sd:.1f} sn |")
    return "\n".join(lines)


def ranking(results, tolerance=TOLERANCE):
    """Seçenekleri val accuracy ortalamasına göre sıralar ve en iyisiyle farkın anlamlı olup olmadığını yazar."""
    rows = []
    for res in results:
        acc, _ = mean_std([r["val_acc"] for r in res["runs"] if not r["failed"]])
        rows.append((res["label"], acc, res["runs"]))
    rows.sort(key=lambda row: -row[1] if math.isfinite(row[1]) else math.inf)
    top = rows[0][1]
    lines = []
    for label, acc, runs in rows:
        if not math.isfinite(acc):
            lines.append(f"{label:<28} {failure_text(runs)}")
            continue
        gap = top - acc
        verdict = "en iyi" if gap == 0 else ("fark anlamlı DEĞİL" if gap < tolerance else "anlamlı şekilde kötü")
        lines.append(f"{label:<28} val acc %{acc * 100:.2f} | en iyiden fark {gap * 100:.2f} puan -> {verdict}")
    return "\n".join(lines)


def best_per_group(results, key):
    """Her grubun (ör. her optimizer'ın) en yüksek val accuracy'li seçeneği; hiç öğrenemeyen grup atlanır."""
    best = {}
    for res in results:
        acc, _ = mean_std([r["val_acc"] for r in res["runs"] if not r["failed"]])
        group = res["config"][key]
        if math.isfinite(acc) and (group not in best or acc > best[group][0]):
            best[group] = (acc, res)
    return [res for _, res in best.values()]


def best_lr_report(results, key, names):
    """Her grubun en iyi learning rate'i ve bu değerin denenen aralığın kenarında olup olmadığı."""
    lines = []
    for res in best_per_group(results, key):
        group, lr = res["config"][key], res["config"]["lr"]
        tried = sorted(r["config"]["lr"] for r in results if r["config"][key] == group)
        acc, _ = mean_std([r["val_acc"] for r in res["runs"] if not r["failed"]])
        # En iyi değer aralığın kenarındaysa, aralığın dışında daha iyi bir değer olabilir
        where = "aralığın KENARINDA" if lr in (tried[0], tried[-1]) else "aralığın içinde"
        lines.append(f"{names[group]:<16} en iyi lr {format_lr(lr):<7} | val acc %{acc * 100:.2f} | "
                     f"denenen {' / '.join(format_lr(v) for v in tried)} -> {where}")
    return "\n".join(lines)


def save(name, results, out_dir=None):
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment = EXPERIMENTS[name]
    payload = {"deney": name, "baslik": experiment["title"], "seeds": list(SEEDS),
               "temel_ayar": {**DEFAULT_CONFIG, **experiment["base"]}, "secenekler": results}
    (out_dir / "sonuclar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list),
                                           encoding="utf-8")
    loss_comparable = experiment.get("plot_metrics") is None
    table = markdown_table(results, loss_comparable)
    curves = results
    if "group_by" in experiment:
        # Learning rate taraması: bütün ayarların grafiği ayrı, eğriler yalnızca her grubun en iyisi için
        fig = plot_lr_sweep(results, experiment["group_by"], experiment["group_names"],
                            title=f"{experiment['title']}: {len(SEEDS)} seed ortalaması ± std")
        fig.savefig(out_dir / "lr_tarama.png")
        plt.close(fig)
        curves = best_per_group(results, experiment["group_by"])
        table += "\n\nEn iyi learning rate'leriyle:\n\n" + markdown_table(curves, loss_comparable)
    (out_dir / "tablo.md").write_text(table + "\n", encoding="utf-8")
    title = experiment["title"] + (" (en iyi learning rate'leriyle)" if "group_by" in experiment else "")
    fig = plot_comparison(curves, title=f"{title}: {len(SEEDS)} seed ortalaması ± std",
                          metrics=experiment.get("plot_metrics"))
    fig.savefig(out_dir / "egriler.png")
    plt.close(fig)
    return out_dir


def main():
    name = sys.argv[1]
    plot_style.apply()
    device = get_device()
    results = run_experiment(name, device)
    experiment = EXPERIMENTS[name]
    loss_comparable = experiment.get("plot_metrics") is None
    print("\n" + markdown_table(results, loss_comparable) + "\n\n" + ranking(results))
    if "group_by" in experiment:
        print("\n" + best_lr_report(results, experiment["group_by"], experiment["group_names"]))
        print("\n" + ranking(best_per_group(results, experiment["group_by"])))
    print("\nKaydedildi:", save(name, results))


if __name__ == "__main__":
    main()
