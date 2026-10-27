"""Aşama 4B–4C: ayarların birlikte aranması ve en iyilerin yeniden ölçülmesi.

4A'da ayarlar tek tek değiştirildi. Bu yöntem, ayarlar arasındaki etkileşimleri (ör. en iyi learning rate'in batch
boyutuna ya da dropout'a bağlı olması) gözden kaçırabilir. 4B'de bütün ayarlar birlikte aranır ve iki arama yöntemi
aynı bütçeyle (aynı deneme sayısı) karşılaştırılır:
- Rastgele arama (Bergstra ve Bengio, 2012): her deneme, arama uzayından bağımsız ve rastgele seçilmiş bir ayar.
- TPE (Bergstra vd., 2011; Optuna kütüphanesi): önceki denemelerin sonuçlarından iyi ve kötü ayarların dağılımını
  öğrenir; sonraki denemeleri umut verici bölgelerden seçer.

Her deneme tek seed (0) ile eğitilir. Çok sayıda denemenin en iyisi, kısmen şanslı bir seed sayesinde en iyi
çıkmış olabilir (seçim yanlılığı). Bu yüzden 4C'de en iyi 5 ayar, 4A'nın sonucuyla birlikte, aramada kullanılmamış
yeni seed'lerle (1, 2, 3) yeniden eğitilir; final ayar bu yeniden ölçüme göre seçilir. Test seti hiç yüklenmez.

Çalıştırma:  python src/arama.py ara       (4B)
             python src/arama.py dogrula   (4C; 4B'nin sonuçlarını okur)
Çıktılar  :  results/arama/denemeler.json, tablo.md, ilerleme.png, ayar_etkileri.png
             results/arama/dogrulama/sonuclar.json, tablo.md, egriler.png
             results/arama/secilen_ayar.json   (Aşama 5'in eğiteceği ayar)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.ticker import FuncFormatter

import plot_style
from augment import AUGMENT_LEVELS
from data import load_data
from device import get_device
from karsilastirma import (BASE_4B, TOLERANCE, cheapest_report, cheapest_within_tolerance, format_count, format_lr,
                          markdown_table, mean_val_acc, parameter_count, ranking, run_configs, summarize_run)
from plots import plot_comparison
from train import DEFAULT_CONFIG, train

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arama"
N_TRIALS = 30          # her yöntem için deneme sayısı (bütçe)
TRIAL_SEED = 0         # aramadaki her eğitimin seed'i
VERIFY_SEEDS = (1, 2, 3)  # 4C: aramada kullanılmamış seed'ler
TOP_K = 5
METHODS = {"rastgele": "Rastgele arama", "tpe": "TPE (Optuna)"}


def suggest_config(trial):
    """Arama uzayı. Aralıklar 4A'nın sonuçlarına göre seçildi; ödev şartları (ReLU, ≥ 2 gizli katman) sabit."""
    depth = trial.suggest_int("depth", 2, 4)
    width = trial.suggest_categorical("width", [128, 256, 512, 1024])
    optimizer = trial.suggest_categorical("optimizer", ["sgd_momentum", "adam"])
    config = {**BASE_4B, "hidden": (width,) * depth, "optimizer": optimizer}
    config["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    if optimizer == "adam":
        config["lr"] = trial.suggest_float("lr_adam", 3e-5, 1e-2, log=True)
    else:
        # 4A-6: SGD'de iyi learning rate batch ile orantılı. Aranan değer batch 64'e göre learning rate'tir;
        # gerçek learning rate = lr_64 × batch / 64 (doğrusal ölçekleme kuralı).
        lr_64 = trial.suggest_float("lr_sgd_64", 0.002, 0.05, log=True)
        config["lr"] = lr_64 * config["batch_size"] / 64
        config["momentum"] = trial.suggest_float("momentum", 0.8, 0.98)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    config["batchnorm"] = trial.suggest_categorical("batchnorm", [False, True])
    config["augment"] = AUGMENT_LEVELS.get(trial.suggest_categorical("augment", ["yok", "hafif", "orta", "guclu"]))
    return config


def make_sampler(method, seed=0):
    # İki yöntem aynı seed'i kullanır. TPE'nin ilk 10 denemesi de rastgele olduğu için iki yöntem aynı 10 denemeyle
    # başlar; 11. denemeden itibaren aradaki fark yalnızca yöntemden gelir (eşleştirilmiş karşılaştırma).
    if method == "rastgele":
        return optuna.samplers.RandomSampler(seed=seed)
    # İlk 10 deneme rastgele: TPE'nin öğrenebilmesi için önce biraz veri gerekir
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)


def describe(config):
    """Bir ayarın kısa, okunur özeti (tablolar için)."""
    config = {**DEFAULT_CONFIG, **config}
    hidden = config["hidden"]
    optimizer = "Adam" if config["optimizer"] == "adam" else f"mSGD (m = {config['momentum']:.2f})"
    lr = float(f"{config['lr']:.2g}")  # 2 anlamlı basamak
    parts = [f"{len(hidden)} × {hidden[0]}", optimizer, f"lr {format_lr(lr)}", f"batch {config['batch_size']}",
             f"dropout {config['dropout']:.2f}", f"wd {config['weight_decay']:.0e}"]
    if config.get("batchnorm"):
        parts.append("BN")
    if config.get("augment"):
        level = next((name for name, value in AUGMENT_LEVELS.items() if value == config["augment"]), "özel")
        parts.append(f"veri artırma {level}")
    return ", ".join(parts)


def search(method, device, n_trials=N_TRIALS, data=None, verbose=True):
    """Bir yöntemle n_trials deneme yapar. Her deneme bir eğitim; hedef, validation accuracy."""
    data = data or load_data(device=device)[0]
    trials = []

    def objective(trial):
        config = suggest_config(trial)
        run = summarize_run(train({**config, "seed": TRIAL_SEED}, data, device, verbose=False))
        trials.append({"yontem": method, "deneme": trial.number + 1, "ayar": config, "parametreler": trial.params,
                       **{k: v for k, v in run.items() if k != "history"}, "history": run["history"]})
        if verbose:
            status = "öğrenemedi" if run["failed"] else f"val acc %{run['val_acc'] * 100:.2f}"
            print(f"{METHODS[method]:<15} deneme {trial.number + 1:>2}: {status} | {run['train_seconds']:5.1f} sn "
                  f"| {describe(config)}")
        return 0.0 if run["failed"] else run["val_acc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=make_sampler(method))
    study.optimize(objective, n_trials=n_trials)
    return trials


def best_so_far(trials):
    values = [0.0 if t["failed"] else t["val_acc"] for t in trials]
    return np.maximum.accumulate(values)


def search_table(trials, top=10):
    """En iyi denemeler (tek seed)."""
    lines = ["| # | Yöntem | Deneme | Ayar | Parametre | Val acc (tek seed) | En iyi epoch / eğitilen | Süre |",
             "|---|---|---|---|---|---|---|---|"]
    ranked = sorted((t for t in trials if not t["failed"]), key=lambda t: -t["val_acc"])[:top]
    for i, t in enumerate(ranked, 1):
        lines.append(f"| {i} | {METHODS[t['yontem']]} | {t['deneme']} | {describe(t['ayar'])} | "
                     f"{format_count(t['parameters'])} | %{t['val_acc'] * 100:.2f} | "
                     f"{t['best_epoch']} / {len(t['history'])} | {t['train_seconds']:.1f} sn |")
    return "\n".join(lines)


def method_summary(trials):
    """Her yöntemin özeti: en iyi deneme, ilk 10 / son 10 denemenin ortalaması, öğrenemeyen deneme sayısı."""
    lines = ["| Yöntem | En iyi val acc | İlk 10 denemenin ortalaması | Son 10 denemenin ortalaması | "
             "Öğrenemeyen deneme | Toplam süre |", "|---|---|---|---|---|---|"]
    for method, name in METHODS.items():
        own = [t for t in trials if t["yontem"] == method]
        values = [0.0 if t["failed"] else t["val_acc"] for t in own]
        lines.append(f"| {name} | %{max(values) * 100:.2f} | %{np.mean(values[:10]) * 100:.2f} | "
                     f"%{np.mean(values[-10:]) * 100:.2f} | {sum(t['failed'] for t in own)} | "
                     f"{sum(t['train_seconds'] for t in own) / 60:.1f} dk |")
    return "\n".join(lines)


def plot_progress(trials, floor=95.0):
    """Her yöntemin denemeleri (noktalar) ve o ana kadarki en iyi sonuç (çizgi)."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (method, name) in enumerate(METHODS.items()):
        own = [t for t in trials if t["yontem"] == method]
        x = np.arange(1, len(own) + 1)
        values = np.array([0.0 if t["failed"] else t["val_acc"] * 100 for t in own])
        color = plot_style.SERIES[i]
        ax.scatter(x, np.maximum(values, floor), s=18, color=color, alpha=0.45, linewidths=0)
        ax.step(x, best_so_far(own) * 100, where="post", color=color, label=f"{name}: o ana kadarki en iyi",
                linestyle=(0, (5, 3)) if i == 0 else "-", zorder=3 if i == 0 else 2)
    ax.set_ylim(floor - 0.1, None)
    ax.set_xlabel("Deneme")
    ax.set_title(f"Validation accuracy (%); %{floor:.0f}'in altındaki denemeler en altta gösterildi", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def augment_level(config):
    return next((i for i, value in enumerate(AUGMENT_LEVELS.values(), 1) if value == config.get("augment")), 0)


def plot_effects(trials, floor=96.0):
    """Her ayarın validation accuracy ile ilişkisi (iki yöntemin bütün denemeleri)."""
    ok = [t for t in trials if not t["failed"]]
    acc = np.array([max(t["val_acc"] * 100, floor) for t in ok])
    # Kategorik ayarlarda noktalar üst üste binmesin diye yatayda biraz kaydırılır
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(ok))
    panels = [
        ("Learning rate", [t["ayar"]["lr"] for t in ok], "log", None),
        ("Katman başına nöron", [t["ayar"]["hidden"][0] for t in ok], "log2", None),
        ("Gizli katman sayısı", [len(t["ayar"]["hidden"]) + j for t, j in zip(ok, jitter)], "category", ["2", "3", "4"]),
        ("Batch boyutu", [t["ayar"]["batch_size"] for t in ok], "log2", None),
        ("Dropout", [t["ayar"]["dropout"] for t in ok], "linear", None),
        ("Weight decay", [t["ayar"]["weight_decay"] for t in ok], "log", None),
        ("Batch normalization", [int(t["ayar"]["batchnorm"]) + j for t, j in zip(ok, jitter)], "category",
         ["yok", "var"]),
        ("Veri artırma", [augment_level(t["ayar"]) + j for t, j in zip(ok, jitter)], "category",
         ["yok", "hafif", "orta", "güçlü"]),
    ]
    colors = [plot_style.SERIES[0] if t["ayar"]["optimizer"] == "sgd_momentum" else plot_style.SERIES[1] for t in ok]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (title, values, scale, labels) in zip(axes.flat, panels):
        ax.scatter(values, acc, s=16, c=colors, alpha=0.7, linewidths=0)
        if scale == "log":
            ax.set_xscale("log")
        elif scale == "log2":
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        elif scale == "category":
            start = 2 if title == "Gizli katman sayısı" else 0
            ax.set_xticks(range(start, start + len(labels)), labels)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(floor - 0.1, None)
    handles = [plt.Line2D([], [], linestyle="none", marker="o", color=plot_style.SERIES[k]) for k in (0, 1)]
    fig.legend(handles, ["Momentumlu SGD", "Adam"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"Validation accuracy (%, tek seed); %{floor:.0f}'nın altı en altta", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def save_search(trials, out_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"deneme_sayisi": N_TRIALS, "seed": TRIAL_SEED, "temel_ayar": {**DEFAULT_CONFIG, **BASE_4B},
               "denemeler": trials}
    (out_dir / "denemeler.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list),
                                            encoding="utf-8")
    (out_dir / "tablo.md").write_text(method_summary(trials) + "\n\nEn iyi 10 deneme:\n\n" + search_table(trials)
                                      + "\n", encoding="utf-8")
    for name, fig in [("ilerleme.png", plot_progress(trials)), ("ayar_etkileri.png", plot_effects(trials))]:
        fig.savefig(out_dir / name)
        plt.close(fig)
    return out_dir


def load_trials(out_dir=RESULTS_DIR):
    trials = json.loads((Path(out_dir) / "denemeler.json").read_text(encoding="utf-8"))["denemeler"]
    for t in trials:
        t["ayar"]["hidden"] = tuple(t["ayar"]["hidden"])
    return trials


def top_configs(trials, k=TOP_K):
    """Tek seed'le en yüksek val accuracy'yi veren k farklı ayar + karşılaştırma için 4A'nın sonucu."""
    configs = {}
    for t in sorted((t for t in trials if not t["failed"]), key=lambda t: -t["val_acc"]):
        if t["ayar"] not in configs.values():
            configs[f"{METHODS[t['yontem']]} #{t['deneme']}"] = t["ayar"]
        if len(configs) == k:
            break
    configs["4A sonucu"] = {k: v for k, v in {**DEFAULT_CONFIG, **BASE_4B}.items() if k != "seed"}
    return configs


def verify(trials, device, seeds=VERIFY_SEEDS, verbose=True):
    """4C: en iyi ayarları aramada kullanılmamış seed'lerle yeniden eğitir."""
    configs = top_configs(trials)
    if verbose:
        for label, config in configs.items():
            print(f"{label:<22} {describe(config)}")
        print()
    return run_configs(configs, device, seeds, verbose)


def choose(results):
    """Seçim kuralı (4A-4 ile aynı): en iyiye 0.3 puandan yakın olanların en az parametrelisi."""
    return cheapest_within_tolerance(results, parameter_count)[0]


def save_verification(results, out_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    verify_dir = out_dir / "dogrulama"
    verify_dir.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": list(VERIFY_SEEDS), "secenekler": results}
    (verify_dir / "sonuclar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list),
                                              encoding="utf-8")
    descriptions = "\n".join(f"- **{res['label']}:** {describe(res['config'])}" for res in results)
    (verify_dir / "tablo.md").write_text(descriptions + "\n\n" + markdown_table(results, show_parameters=True) + "\n",
                                         encoding="utf-8")
    fig = plot_comparison(results, title=f"En iyi ayarlar, yeni seed'lerle: {len(VERIFY_SEEDS)} seed ortalaması ± std")
    fig.savefig(verify_dir / "egriler.png")
    plt.close(fig)
    chosen = choose(results)
    config = {k: v for k, v in chosen["config"].items() if k != "seed"}
    selected = {"kaynak": chosen["label"], "aciklama": describe(config), "val_acc_ortalama": mean_val_acc(chosen),
                "seeds": list(VERIFY_SEEDS), "ayar": config}
    (out_dir / "secilen_ayar.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False, default=list),
                                               encoding="utf-8")
    return out_dir


def main():
    step = sys.argv[1]
    plot_style.apply()
    device = get_device()
    if step == "ara":
        data = load_data(device=device)[0]
        trials = [t for method in METHODS for t in search(method, device, data=data)]
        print("\n" + method_summary(trials) + "\n\n" + search_table(trials))
        print("\nKaydedildi:", save_search(trials))
    elif step == "dogrula":
        results = verify(load_trials(), device)
        print("\n" + markdown_table(results, show_parameters=True) + "\n\n" + ranking(results))
        print("\n" + cheapest_report(results, parameter_count))
        print("\nKaydedildi:", save_verification(results))
    else:
        raise SystemExit("Kullanım: python src/arama.py ara | dogrula")


if __name__ == "__main__":
    main()
