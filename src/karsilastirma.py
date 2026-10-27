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
import torch

import plot_style
from augment import AUGMENT_LEVELS
from data import load_data
from device import get_device
from epoch_secimi import policy_report, save_policy_analysis
from plots import plot_comparison, plot_generalization_gap, plot_sweep
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
# Karar 4A-4 (README 5.4): 2 gizli katman × 256 nöron (baseline: 256 → 128).
BASE_4A5 = {**BASE_4A4, "hidden": (256, 256)}
# Karar 4A-5 (README 5.5): ReLU (ödev şartı). Baseline'daki ayar korundu.
BASE_4A6 = {**BASE_4A5, "activation": "relu"}
# Karar 4A-6 (README 5.6): batch 512, lr = 0.08 (lr batch ile orantılı; baseline: batch 64, lr = 0.01).
BASE_4A7 = {**BASE_4A6, "batch_size": 512, "lr": 0.08}
# Karar 4A-7 (README 5.7): veri artırma, hafif güçte (baseline: regularization yok).
BASE_4A8 = {**BASE_4A7, "augment": AUGMENT_LEVELS["hafif"]}
# Karar 4A-8 (README 5.8): early stopping (val loss, patience 5, en fazla 100 epoch; baseline: sabit 20 epoch).
# 4A'nın sonucu: Aşama 4B'deki aramanın başlangıç noktası.
BASE_4B = {**BASE_4A8, "epochs": 100, "patience": 5, "select_by": "val_loss"}

# 4A-3: her optimizer kendine uygun bir learning rate aralığında denenir. Değerler ~3 kat aralıklı.
# Momentumlu SGD'nin aralığı düz SGD'ninkinin 1/10'u: momentum 0.9 iken etkin adım ~lr / (1 − 0.9) = 10 × lr.
# Adam'ın aralığı bir değer daha aşağı uzatıldı: ön denemede en iyi Adam değeri aralığın en küçüğü (0.0001) çıktı.
OPTIMIZER_NAMES = {"sgd": "SGD", "sgd_momentum": "Momentumlu SGD", "adam": "Adam"}
LR_GRID = {
    "sgd": (0.01, 0.03, 0.1, 0.3),
    "sgd_momentum": (0.001, 0.003, 0.01, 0.03),
    "adam": (0.00003, 0.0001, 0.0003, 0.001, 0.003),
}

# 4A-4: derinlik × genişlik ızgarası; her gizli katmanda aynı sayıda nöron.
# Ödev en az 2 gizli katman istiyor; 1 gizli katman yalnızca karşılaştırma için denenir, seçilemez.
DEPTHS = (1, 2, 3, 4)
WIDTHS = (64, 128, 256, 512, 1024)
MIN_DEPTH = 2

# 4A-5: aktivasyon fonksiyonu. Ödev ReLU istiyor; Sigmoid ve Tanh, ReLU'nun neden tercih edildiğini görmek için
# denenir. Aktivasyonların gradyan büyüklükleri farklı olduğu için her biri kendi en iyi learning rate'iyle karşılaştırılır.
# Sigmoid'in aralığı bir değer yukarı uzatıldı: ön denemede en iyi Sigmoid değeri aralığın en büyüğü (0.1) çıktı.
ACTIVATION_NAMES = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh"}
ACTIVATION_LRS = {
    "relu": (0.003, 0.01, 0.03, 0.1),
    "sigmoid": (0.003, 0.01, 0.03, 0.1, 0.3),
    "tanh": (0.003, 0.01, 0.03, 0.1),
}

# 4A-6: batch boyutu, iki learning rate kuralıyla: lr sabit ya da batch ile orantılı
# (lr = 0.01 × B / 64, "doğrusal ölçekleme kuralı"; Goyal vd., 2017). B = 64'te iki kural aynı ayarı verir.
# Aralık iki değer yukarı uzatıldı: ön denemede seçilen (en hızlı, eşit derecede iyi) ayar aralığın en büyüğü (512) çıktı.
BATCH_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048)
LR_RULE_NAMES = {"sabit": "lr sabit", "orantili": "lr batch ile orantılı"}

# 4A-7: regularization yöntemleri; her biri tek başına ve farklı güçlerde. Veri artırma (augment.py) da bir
# regularization yöntemi: model her epoch aynı rakamın biraz döndürülmüş, kaydırılmış, ölçeklenmiş bir hâlini görür.
REGULARIZATIONS = {
    "Yok": {},
    "Dropout 0.1": {"dropout": 0.1},
    "Dropout 0.2": {"dropout": 0.2},
    "Dropout 0.5": {"dropout": 0.5},
    "Weight decay 0.0001": {"weight_decay": 0.0001},
    "Weight decay 0.0005": {"weight_decay": 0.0005},
    "Weight decay 0.002": {"weight_decay": 0.002},
    "Batch normalization": {"batchnorm": True},
    "Veri artırma (hafif)": {"augment": AUGMENT_LEVELS["hafif"]},
    "Veri artırma (orta)": {"augment": AUGMENT_LEVELS["orta"]},
    "Veri artırma (güçlü)": {"augment": AUGMENT_LEVELS["guclu"]},
}
REGULARIZATION_NAMES = {"yok": "Yok", "dropout": "Dropout", "weight_decay": "Weight decay",
                        "batchnorm": "Batch normalization", "augment": "Veri artırma"}

# 4A-8: uzun bir eğitim (100 epoch) bir kez yapılır; farklı epoch / early stopping / seçim ölçütü kuralları bu
# eğitimin geçmişi üzerinde uygulanır (epoch_secimi.py). Validation seti ikiye bölünür: yarısı seçim için,
# diğer yarısı (holdout) seçilen epoch'un gerçekten ne kadar iyi olduğunu ölçmek için.
LONG_EPOCHS = 100


def format_lr(lr):
    """Learning rate'i bilimsel gösterim (3e-05) yerine ondalık sayı (0.00003) olarak yazar."""
    return np.format_float_positional(lr, trim="-")


def regularization_type(config):
    if config.get("augment"):
        return "augment"
    if config.get("batchnorm"):
        return "batchnorm"
    if config.get("dropout", 0) > 0:
        return "dropout"
    if config.get("weight_decay", 0) > 0:
        return "weight_decay"
    return "yok"


def eligible_depth(config):
    """Ödev şartı: en az 2 gizli katman."""
    return len(config["hidden"]) >= MIN_DEPTH


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
        "group": {"of": lambda config: config["optimizer"], "names": OPTIMIZER_NAMES,
                  "best_title": "en iyi learning rate'leriyle"},
        "sweep": {"x": lambda config: config["lr"], "x_label": "Learning rate (logaritmik eksen)",
                  "file": "lr_tarama.png"},
        "report": lambda results: best_lr_report(results, "optimizer", OPTIMIZER_NAMES),
    },
    "4a4_derinlik_genislik": {
        "title": "Derinlik ve genişlik",
        "base": BASE_4A4,
        "variants": {f"{depth} katman × {width} nöron": {"hidden": (width,) * depth}
                     for depth in DEPTHS for width in WIDTHS},
        "show_parameters": True,
        "group": {"of": lambda config: len(config["hidden"]),
                  "names": {depth: f"{depth} gizli katman" + (" (yalnızca karşılaştırma)" if depth < MIN_DEPTH else "")
                            for depth in DEPTHS},
                  "best_title": "her derinliğin en iyisi"},
        "sweep": {"x": lambda config: config["hidden"][0], "x_label": "Katman başına nöron sayısı (logaritmik eksen)",
                  "log_base": 2, "file": "izgara.png"},
        # Seçim: ödev şartını sağlayan ve en iyiye eşikten yakın modellerden en az parametreli olanı
        "report": lambda results: cheapest_report(results, parameter_count, eligible_depth),
    },
    "4a5_aktivasyon": {
        "title": "Aktivasyon fonksiyonu",
        "base": BASE_4A5,
        "variants": {f"{ACTIVATION_NAMES[act]}, lr = {format_lr(lr)}": {"activation": act, "lr": lr}
                     for act, lrs in ACTIVATION_LRS.items() for lr in lrs},
        "group": {"of": lambda config: config["activation"], "names": ACTIVATION_NAMES,
                  "best_title": "en iyi learning rate'leriyle"},
        "sweep": {"x": lambda config: config["lr"], "x_label": "Learning rate (logaritmik eksen)",
                  "file": "lr_tarama.png"},
        "report": lambda results: best_lr_report(results, "activation", ACTIVATION_NAMES),
    },
    "4a6_batch": {
        "title": "Batch boyutu",
        "base": BASE_4A6,
        "variants": {
            **{f"Batch {b}, lr = {format_lr(BASE_4A6['lr'])}": {"batch_size": b} for b in BATCH_SIZES},
            **{f"Batch {b}, lr = {format_lr(BASE_4A6['lr'] * b / 64)}": {"batch_size": b, "lr": BASE_4A6["lr"] * b / 64}
               for b in BATCH_SIZES if b != 64},
        },
        "group": {"of": lambda config: "sabit" if config["lr"] == BASE_4A6["lr"] else "orantili",
                  "names": LR_RULE_NAMES, "best_title": "her kuralın en iyisi"},
        "sweep": {"x": lambda config: config["batch_size"], "x_label": "Batch boyutu (logaritmik eksen)",
                  "log_base": 2, "file": "batch_tarama.png"},
        # Seçim: en iyiye eşikten yakın olanlardan en kısa sürede eğitileni
        "report": lambda results: cheapest_report(results, mean_train_seconds),
    },
    "4a7_regularization": {
        "title": "Regularization",
        "base": BASE_4A7,
        "variants": REGULARIZATIONS,
        "group": {"of": regularization_type, "names": REGULARIZATION_NAMES, "best_title": "her yöntemin en iyisi"},
        "save_extra": lambda results, out_dir: save_gap_plot(results, out_dir, regularization_type),
    },
    "4a8_epoch": {
        "title": "Epoch sayısı, early stopping ve model seçme ölçütü",
        "base": {**BASE_4A8, "epochs": LONG_EPOCHS},
        "variants": {f"{LONG_EPOCHS} epoch": {}},
        "holdout": True,
        "report": lambda results: policy_report(results, TOLERANCE),
        "save_extra": lambda results, out_dir: save_policy_analysis(results, out_dir, TOLERANCE),
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
        "parameters": result["n_params"],
        "history": history,
    }


def failure_text(runs):
    if any(r["diverged"] for r in runs):
        return "ıraksadı (loss NaN)"
    return f"öğrenemedi (hep aynı tahmin, val acc %{runs[0]['val_acc'] * 100:.2f})"


def run_experiment(name, device, seeds=SEEDS, verbose=True, only=None):
    """Bir deneyin bütün seçeneklerini her seed ile eğitir.

    only: yalnızca bu ayarlara sahip seçenekleri eğitir. Ayar sözlüğü (ör. {"optimizer": "sgd"}) ya da
          ayarları alıp True/False döndüren bir fonksiyon (ör. lambda config: len(config["hidden"]) == 2).
    """
    experiment = EXPERIMENTS[name]
    configs = {}
    for label, change in experiment["variants"].items():
        config = {**experiment["base"], **change}
        if only and not (only(config) if callable(only) else all(config.get(k) == v for k, v in only.items())):
            continue
        configs[label] = config
    return run_configs(configs, device, seeds, verbose, holdout=experiment.get("holdout", False))


def run_configs(configs, device, seeds=SEEDS, verbose=True, holdout=False):
    """{etiket: ayar} sözlüğündeki her ayarı her seed ile eğitir. Test seti yüklenmez.

    holdout=True: validation seti ikiye bölünür (split_holdout).
    """
    data_cache = {}
    results = []
    for label, config in configs.items():
        normalization = config.get("normalization", DEFAULT_CONFIG["normalization"])
        if normalization not in data_cache:
            data_cache[normalization], _ = load_data(normalization=normalization, device=device)  # test yüklenmez
            if holdout:
                data_cache[normalization] = split_holdout(data_cache[normalization])
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


def split_holdout(data, seed=0):
    """Validation setini rastgele ikiye böler: "val" (model seçimi için) ve "holdout" (seçimi değerlendirmek için)."""
    x, y = data["val"]
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).to(x.device)
    half = len(x) // 2
    return {**data, "val": (x[perm[:half]], y[perm[:half]]), "holdout": (x[perm[half:]], y[perm[half:]])}


def mean_std(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def format_count(n):
    """Büyük sayıyı üçerli gruplar hâlinde yazar: 235146 -> "235 146"."""
    return f"{n:,}".replace(",", " ")


def markdown_table(results, loss_comparable=True, show_parameters=False):
    """Seçeneklerin karşılaştırma tablosu (3 seed ortalaması ± std, en düşük val loss'lu epoch'ta).

    loss_comparable=False: seçenekler farklı loss fonksiyonları kullanıyorsa loss sütunu kıyaslanamaz.
    show_parameters=True: modelin parametre sayısı sütunu eklenir (ör. farklı model boyutları).
    """
    epochs = results[0]["config"].get("epochs", DEFAULT_CONFIG["epochs"])
    loss_header = "Val loss (en düşük)" if loss_comparable else "Val loss (her biri kendi loss'u, kıyaslanamaz)"
    parameter_header = " Parametre |" if show_parameters else ""
    lines = [f"| Seçenek |{parameter_header} Val acc | {loss_header} | 1. epoch val acc | Train acc | En iyi epoch | "
             f"Eğitim süresi ({epochs} epoch) |",
             "|---|---|---|---|---|---|---|" + ("---|" if show_parameters else "")]
    for res in results:
        runs = [r for r in res["runs"] if not r["failed"]]
        parameters = f" {format_count(res['runs'][0]['parameters'])} |" if show_parameters else ""
        if not runs:
            lines.append(f"| {res['label']} |{parameters} {failure_text(res['runs'])} | – | – | – | – | – |")
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
        lines.append(f"| {label} |{parameters} %{acc * 100:.2f} ± {acc_sd * 100:.2f} | {loss:.4f} ± {loss_sd:.4f} | "
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


def best_per_group(results, group):
    """Her grubun (ör. her optimizer'ın) en yüksek val accuracy'li seçeneği; hiç öğrenemeyen grup atlanır.

    group: ayar adı (ör. "optimizer") ya da ayarlardan grubu bulan fonksiyon.
    """
    group_of = group if callable(group) else lambda config: config[group]
    best = {}
    for res in results:
        acc, _ = mean_std([r["val_acc"] for r in res["runs"] if not r["failed"]])
        key = group_of(res["config"])
        if math.isfinite(acc) and (key not in best or acc > best[key][0]):
            best[key] = (acc, res)
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


def mean_val_acc(res):
    return mean_std([r["val_acc"] for r in res["runs"] if not r["failed"]])[0]


def parameter_count(res):
    return res["runs"][0]["parameters"]


def mean_train_seconds(res):
    return mean_std([r["train_seconds"] for r in res["runs"] if not r["failed"]])[0]


def cheapest_within_tolerance(results, cost, eligible=None, tolerance=TOLERANCE):
    """Seçilebilir seçeneklerden en iyisine eşikten yakın olanlar, maliyete göre ucuzdan pahalıya.

    İlk eleman seçilen seçenektir: doğruluk farkı gürültü düzeyindeyse pahalı seçeneğin daha iyi olduğuna dair
    kanıt yoktur; bu durumda daha ucuz olan tercih edilir.
    cost: bir seçeneğin maliyeti (ör. parameter_count ya da mean_train_seconds).
    eligible: ayarları alıp seçenek seçilebilir mi diye True/False döndüren fonksiyon (ör. ödev şartı).
    """
    scored = [(mean_val_acc(res), res) for res in results if eligible is None or eligible(res["config"])]
    top = max(acc for acc, _ in scored if math.isfinite(acc))
    # Maliyeti eşit olanlar arasında (ör. aynı parametre sayısı) doğruluğu yüksek olan önce gelir
    return sorted((res for acc, res in scored if top - acc < tolerance), key=lambda res: (cost(res), -mean_val_acc(res)))


def cheapest_report(results, cost, eligible=None, tolerance=TOLERANCE):
    """cheapest_within_tolerance'ın adayları; ilk satır seçilen seçenek."""
    candidates = cheapest_within_tolerance(results, cost, eligible, tolerance)
    top = max(mean_val_acc(res) for res in candidates)
    cheapest_tried = min(cost(res) for res in results if eligible is None or eligible(res["config"]))
    lines = [f"Seçilebilir seçeneklerin en iyisinden {tolerance * 100:.1f} puandan az geride olanlar "
             f"(ucuzdan pahalıya):"]
    for i, res in enumerate(candidates):
        note = "  <- SEÇİLEN" if i == 0 else ""
        if i == 0 and cost(res) == cheapest_tried:
            note += " (denenenlerin en ucuzu: daha ucuz bir seçenek de yetebilir)"
        lines.append(f"  {res['label']:<28} {format_count(parameter_count(res)):>9} parametre | "
                     f"val acc %{mean_val_acc(res) * 100:.2f} | en iyiden fark {(top - mean_val_acc(res)) * 100:.2f} "
                     f"puan | {mean_train_seconds(res):.1f} sn{note}")
    return "\n".join(lines)


def save_gap_plot(results, out_dir, group_of):
    """Her grubun en iyisi için train–validation loss farkı grafiği (fark.png)."""
    fig = plot_generalization_gap(best_per_group(results, group_of))
    fig.savefig(Path(out_dir) / "fark.png")
    plt.close(fig)


def save(name, results, out_dir=None):
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment = EXPERIMENTS[name]
    payload = {"deney": name, "baslik": experiment["title"], "seeds": list(SEEDS),
               "temel_ayar": {**DEFAULT_CONFIG, **experiment["base"]}, "secenekler": results}
    (out_dir / "sonuclar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list),
                                           encoding="utf-8")
    loss_comparable = experiment.get("plot_metrics") is None
    show_parameters = experiment.get("show_parameters", False)
    table = markdown_table(results, loss_comparable, show_parameters)
    curves = results
    title = experiment["title"]
    group, sweep = experiment.get("group"), experiment.get("sweep")
    if sweep:
        # Izgara taraması: her grup bir çizgi, x ekseninde taranan değer
        fig = plot_sweep(results, group["of"], sweep["x"], group["names"], sweep["x_label"],
                         title=f"{title}: {len(SEEDS)} seed ortalaması ± std", log_base=sweep.get("log_base", 10))
        fig.savefig(out_dir / sweep["file"])
        plt.close(fig)
    if group:
        # Seçenek çok olduğunda eğriler yalnızca her grubun en iyisi için çizilir
        curves = best_per_group(results, group["of"])
        best_title = group["best_title"]
        table += (f"\n\n{best_title[0].upper() + best_title[1:]}:\n\n"
                  + markdown_table(curves, loss_comparable, show_parameters))
        title += f" ({best_title})"
    (out_dir / "tablo.md").write_text(table + "\n", encoding="utf-8")
    fig = plot_comparison(curves, title=f"{title}: {len(SEEDS)} seed ortalaması ± std",
                          metrics=experiment.get("plot_metrics"))
    fig.savefig(out_dir / "egriler.png")
    plt.close(fig)
    if "save_extra" in experiment:
        experiment["save_extra"](results, out_dir)
    return out_dir


def main():
    name = sys.argv[1]
    plot_style.apply()
    device = get_device()
    results = run_experiment(name, device)
    experiment = EXPERIMENTS[name]
    loss_comparable = experiment.get("plot_metrics") is None
    print("\n" + markdown_table(results, loss_comparable, experiment.get("show_parameters", False))
          + "\n\n" + ranking(results))
    if "group" in experiment:
        print("\n" + ranking(best_per_group(results, experiment["group"]["of"])))
    if "report" in experiment:
        print("\n" + experiment["report"](results))
    print("\nKaydedildi:", save(name, results))


if __name__ == "__main__":
    main()
