"""Aşama 6: kendi el yazımızla yazılmış rakamları MNIST formatına getirip modele vermek.

Akış: fotoğraf → gri ton → rakamları bulma (her rakam ayrı bir mürekkep parçası) → her rakamı 28 × 28'lik
MNIST formatına getirme → train istatistikleriyle standardizasyon → tahmin.

Üç ön işleme karşılaştırılır:
- "ham"       : rakamın kutusu doğrudan 28 × 28'e küçültülür (renkler ters değil: beyaz kâğıt, koyu mürekkep)
- "ters_renk" : aynısı, ama renkler ters çevrilir (MNIST'teki gibi siyah zemin, beyaz rakam)
- "mnist"     : MNIST'in kendi hazırlanma adımları (LeCun vd., 1998): kâğıt arka planı temizlenir, çizgi
                kalınlığı MNIST'e yaklaştırılır, rakam oranı korunarak 20 × 20'lik kutuya sığdırılır ve
                28 × 28'lik görüntüde kütle merkezi ortaya gelecek şekilde yerleştirilir.

Veri artırmanın etkisini görmek için final ayarın veri artırma ayarı tersine çevrilmiş bir kopyası da eğitilir ve
iki model hem kendi rakamlarımızda hem de değiştirilmiş (döndürülmüş, kaydırılmış, ölçeklenmiş) validation
görüntülerinde karşılaştırılır. Test seti kullanılmaz.

Çalıştırma:  python src/el_yazisi.py <öğrenci_numarası>   (fotoğraflardaki bütün rakamlar sırayla; ör. iki fotoğrafta
             numara iki kez yazıldıysa numara iki kez)
Girdi     :  el_yazisi/ham/ içindeki fotoğraflar (dosya adı sırasıyla; her fotoğrafta satır satır, soldan sağa)
Çıktılar  :  el_yazisi/islenmis/*.png (MNIST formatındaki 28 × 28 rakamlar), results/el_yazisi/
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

import plot_style
from augment import fixed_transform

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "el_yazisi" / "ham"
PROCESSED_DIR = ROOT / "el_yazisi" / "islenmis"
RESULTS_DIR = ROOT / "results" / "el_yazisi"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PIPELINES = {"ham": "Ham (yalnızca küçültme)", "ters_renk": "Ters renk + küçültme", "mnist": "MNIST tarzı"}

# MNIST'te rakamlar 20 × 20'lik bir kutuya sığdırılmış; çizgi kalınlığı rakam yüksekliğinin ~%10'u kadar
MNIST_BOX = 20
TARGET_STROKE_RATIO = 0.10


def read_gray(path):
    """Fotoğrafı gri tonlu (0 = siyah, 255 = beyaz) okur.

    cv2.imread, Türkçe karakter içeren Windows yollarını açamadığı için dosya önce bayt olarak okunur.
    """
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return image


def list_photos(folder=RAW_DIR):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def ink_mask(gray):
    """Mürekkep olan pikseller 255, kâğıt 0. Işık fotoğraf boyunca değişebildiği için eşik yerel hesaplanır."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(31, (min(gray.shape) // 8) | 1)  # tek sayı olmalı; fotoğraf büyüdükçe komşuluk da büyür
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15)
    # Küçük lekeleri sil, kopuk çizgileri birleştir
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def paper_region(gray, min_contrast=40):
    """Fotoğrafta kâğıdın kapladığı bölge (255 = kâğıt). Kâğıdın dışı (masa, halı) kadraja girdiyse onu ayırır.

    Görüntü küçültülüp güçlü bir median filtreyle bulanıklaştırılır; ince rakam çizgileri ve doku kaybolur. Otsu
    eşiği açık (kâğıt) ve koyu (arka plan) bölgeleri ayırır; en büyük açık bölgenin dışbükey zarfı kâğıt kabul edilir.
    İki bölgenin parlaklık farkı küçükse ayrı bir arka plan yok demektir (ör. bütün kare kâğıt): None döner.
    """
    scale = 512 / max(gray.shape)
    small = cv2.medianBlur(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), 15)
    _, bright = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bright.all() or not bright.any() or small[bright > 0].mean() - small[bright == 0].mean() < min_contrast:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hull = cv2.convexHull(np.argwhere(labels == largest)[:, ::-1].astype(np.int32))
    region = np.zeros_like(small)
    cv2.fillConvexPoly(region, hull, 255)
    return cv2.resize(region, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def find_digits(gray):
    """Fotoğraftaki rakamların kutularını (x, y, genişlik, yükseklik) satır satır, soldan sağa sıralı döner.

    Her rakam ayrı bir mürekkep parçası (bağlı bileşen) kabul edilir. Görüntünün kenarına değen ya da kâğıdın dışında
    kalan parçalar (kâğıdın kenarı, kadraja giren masa ya da halı) atılır. Gürültüyü elemek için en büyük parçaya
    göre çok küçük olanlar da atılır. Kalan parçalar satırlara ayrılır; bir satırda yatayda büyük ölçüde üst üste
    binen parçalar (ör. 5'in kopuk üst çizgisi) birleştirilir.
    """
    mask = ink_mask(gray)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, count)]
    boxes = [(x, y, w, h) for x, y, w, h in boxes if x > 0 and y > 0 and x + w < width and y + h < height]
    paper = paper_region(gray)
    if paper is not None:
        boxes = [(x, y, w, h) for x, y, w, h in boxes if paper[y + h // 2, x + w // 2]]
    if not boxes:
        return []
    tallest = max(h for _, _, _, h in boxes)
    boxes = [b for b in boxes if b[3] >= 0.35 * tallest and b[2] * b[3] >= 0.02 * tallest ** 2]
    return [box for line in group_lines(boxes) for box in merge_overlapping(sorted(line, key=lambda b: b[0]))]


def group_lines(boxes):
    """Kutuları satırlara ayırır (yukarıdan aşağı). Dikeyde yarısından fazlası çakışan kutular aynı satırda sayılır."""
    lines = []
    for box in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        if lines:
            top = min(b[1] for b in lines[-1])
            bottom = max(b[1] + b[3] for b in lines[-1])
            if min(bottom, box[1] + box[3]) - max(top, box[1]) > 0.5 * box[3]:
                lines[-1].append(box)
                continue
        lines.append([box])
    return lines


def merge_overlapping(boxes):
    """Soldan sağa sıralı kutulardan yatayda büyük ölçüde üst üste binenleri birleştirir."""
    merged = []
    for box in boxes:
        if merged:
            x, y, w, h = merged[-1]
            overlap = min(x + w, box[0] + box[2]) - max(x, box[0])
            if overlap > 0.5 * min(w, box[2]):
                x2, y2 = max(x + w, box[0] + box[2]), max(y + h, box[1] + box[3])
                x, y = min(x, box[0]), min(y, box[1])
                merged[-1] = (x, y, x2 - x, y2 - y)
                continue
        merged.append(box)
    return merged


def crop(gray, box, margin=0.15):
    """Kutuyu her yönde kenar payıyla keser (rakam kutunun kenarına yapışmasın).

    Rakamlar birbirine yakınsa kenar payına komşu rakamın parçaları düşebilir. Bu yüzden kutunun dışında kalan kenar
    payı kâğıt rengiyle (parçanın ortanca parlaklığı; parçanın çoğu kâğıt) örtülür.
    """
    x, y, w, h = box
    pad = int(margin * max(w, h))
    y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
    piece = gray[y0:y1, x0:x1].copy()
    outside = np.ones(piece.shape, dtype=bool)
    outside[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    piece[outside] = np.median(piece)
    return piece


def preprocess_raw(digit_crop):
    """En basit yol: kesilen parça doğrudan 28 × 28'e küçültülür. Renkler fotoğraftaki gibi kalır."""
    return cv2.resize(digit_crop, (28, 28), interpolation=cv2.INTER_AREA)


def preprocess_inverted(digit_crop):
    """Küçültme + ters renk: MNIST'teki gibi koyu zemin, açık rakam."""
    return 255 - preprocess_raw(digit_crop)


def stroke_width(binary):
    """Çizgi kalınlığının yaklaşık değeri (piksel).

    Uzaklık dönüşümü her mürekkep pikselinin en yakın kenara uzaklığıdır. w kalınlığında bir çizgide bu değer
    kenarda 0, ortada w/2'dir; ortalaması ~w/4. Bu yüzden kalınlık ≈ 4 × ortalama uzaklık.
    """
    distances = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return 4 * distances[binary > 0].mean()


def preprocess_mnist(digit_crop):
    """MNIST'in hazırlanma adımlarını izleyen ön işleme."""
    # 1) Ters renk ve arka plan temizliği: kâğıt tam siyah (0), mürekkep açık
    inverted = 255 - digit_crop.astype(np.float32)
    # Kesilen küçük parçada ışık hemen hemen eşit; tek bir eşik (Otsu) kalın çizgilerin içini de mürekkep sayar.
    # (Yerel eşik, çizgi komşuluk boyutundan kalınsa çizginin ortasını kâğıt sanıp rakamı içi boş bırakırdı.)
    _, mask = cv2.threshold(cv2.GaussianBlur(digit_crop, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    paper = np.median(inverted[mask == 0]) if (mask == 0).any() else 0.0
    ink = np.percentile(inverted[mask > 0], 90) if (mask > 0).any() else 255.0
    image = np.clip((inverted - paper) / max(ink - paper, 1.0), 0, 1) * 255
    image[cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0] = 0  # mürekkep dışındaki gölge ve lekeler

    # 2) Rakamın sıkı kutusu
    ys, xs = np.nonzero(image > 30)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 3) Çizgi kalınlığı: 20 piksellik kutuya küçültüldükten sonra ~2 piksel olacak şekilde kalınlaştır
    target = TARGET_STROKE_RATIO * max(image.shape)
    width = stroke_width(image > 127)
    if width < target:
        size = int(round(target - width)) | 1
        image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))

    # 4) Oranı koruyarak 20 × 20'lik kutuya sığdır (alan ortalamalı küçültme gri kenarlar üretir, MNIST'teki gibi)
    scale = MNIST_BOX / max(image.shape)
    new_size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    small = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # 5) 28 × 28'lik siyah görüntünün ortasına koy, sonra kütle merkezini (14, 14)'e kaydır (MNIST'te ortalama
    #    kütle merkezi tam bu nokta)
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant", cval=0)
    return np.clip(canvas, 0, 255).astype(np.uint8)


PREPROCESSORS = {"ham": preprocess_raw, "ters_renk": preprocess_inverted, "mnist": preprocess_mnist}


def extract_digits(photos):
    """Bütün fotoğraflardaki rakamları fotoğraf sırasıyla, her fotoğrafta satır satır ve soldan sağa bulur.

    Döner: [(fotoğraf adı, kutu, kesilen gri parça)], fotoğraflar sözlüğü {ad: gri görüntü}.
    """
    digits, grays = [], {}
    for path in photos:
        gray = read_gray(path)
        grays[path.name] = gray
        for box in find_digits(gray):
            digits.append((path.name, box, crop(gray, box)))
    return digits, grays


def to_model_input(images_uint8, mean, std, device="cpu"):
    """0–255 arası 28 × 28 görüntüleri modelin beklediği hâle getirir: [0, 1] ölçek + train istatistikleriyle standardizasyon."""
    x = torch.tensor(np.stack(images_uint8), dtype=torch.float32, device=device).reshape(len(images_uint8), -1)
    return (x / 255.0 - mean) / std


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x).softmax(dim=1).cpu()


def save_processed(images, student_number, folder=PROCESSED_DIR):
    """MNIST formatındaki rakamları teslim için PNG olarak kaydeder: 01_rakam-2.png, 02_rakam-0.png, ..."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for i, image in enumerate(images):
        label = student_number[i] if i < len(student_number) else "x"
        ok, encoded = cv2.imencode(".png", image)
        encoded.tofile(str(folder / f"{i + 1:02d}_rakam-{label}.png"))
    return folder


def plot_digits_table(rows, row_titles, labels, predictions=None, title=None):
    """Satırlar ön işleme yöntemleri, sütunlar rakamlar. Tahmin verilirse her görüntünün altına yazılır."""
    n = len(rows[0])
    fig, axes = plt.subplots(len(rows), n, figsize=(max(n * 0.95, 4), len(rows) * 1.35 + 0.4), squeeze=False)
    for r, (images, row_title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(images[c], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if predictions is not None:
                pred, conf = predictions[r][c]
                correct = labels[c] is not None and str(pred) == labels[c]
                ax.set_xlabel(f"{pred} (%{conf * 100:.0f})", fontsize=8,
                              color=plot_style.SERIES[2] if correct else plot_style.SERIES[7])
            if c == 0:
                ax.set_ylabel(row_title, fontsize=8, rotation=0, ha="right", va="center",
                              color=plot_style.INK_SECONDARY)
            if r == 0 and labels[c] is not None:
                ax.set_title(labels[c], fontsize=10, color=plot_style.INK)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=plot_style.INK)
    fig.tight_layout()
    return fig


def plot_detection(grays, digits):
    """Her fotoğrafta bulunan rakam kutuları ve sıra numaraları (rakamların doğru bulunduğunu kontrol etmek için)."""
    names = list(grays)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)), squeeze=False)
    index = 0
    for ax, name in zip(axes[:, 0], names):
        ax.imshow(grays[name], cmap="gray", vmin=0, vmax=255)
        for photo, (x, y, w, h), _ in digits:
            if photo != name:
                continue
            index += 1
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=plot_style.SERIES[1], linewidth=2))
            ax.text(x, y - 8, str(index), color=plot_style.SERIES[1], fontsize=11, fontweight="semibold")
        ax.set_title(name, loc="left", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def evaluate_pipelines(model, mean, std, digits, student_number, device="cpu"):
    """Her ön işleme yöntemiyle bütün rakamların tahmini; doğruluk ve ortalama güven."""
    labels = list(student_number)
    results = {}
    for name, preprocess in PREPROCESSORS.items():
        images = [preprocess(c) for _, _, c in digits]
        probabilities = predict(model, to_model_input(images, mean, std, device))
        predicted = probabilities.argmax(dim=1).tolist()
        confidence = probabilities.max(dim=1).values.tolist()
        correct = [str(p) == t for p, t in zip(predicted, labels)]
        results[name] = {"images": images, "predicted": predicted, "confidence": confidence,
                         "probabilities": probabilities.tolist(), "correct": correct,
                         "accuracy": sum(correct) / len(correct) if correct else float("nan")}
    return results


def summary_table(results_by_model):
    """{model adı: evaluate_pipelines sonucu} → markdown tablo (satır: ön işleme, sütun: model)."""
    models = list(results_by_model)
    lines = ["| Ön işleme | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name, title in PIPELINES.items():
        cells = []
        for model_name in models:
            r = results_by_model[model_name][name]
            cells.append(f"{sum(r['correct'])}/{len(r['correct'])} doğru (%{r['accuracy'] * 100:.0f}), "
                         f"ortalama güven %{np.mean(r['confidence']) * 100:.0f}")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(results_by_model, student_number, out_dir=RESULTS_DIR, robustness=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"ogrenci_numarasi": student_number,
               "sonuclar": {m: {p: {k: v for k, v in r.items() if k != "images"} for p, r in res.items()}
                            for m, res in results_by_model.items()}}
    (out_dir / "sonuclar.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "tablo.md").write_text(summary_table(results_by_model) + "\n", encoding="utf-8")
    if robustness:
        (out_dir / "dayaniklilik.md").write_text(robustness + "\n", encoding="utf-8")
    return out_dir


# Modeli kendi el yazımızda görebileceğimiz türden değişikliklerle sınamak için (validation setine uygulanır)
ROBUSTNESS_TRANSFORMS = {
    "Orijinal": {},
    "15° döndürülmüş": {"rotation": 15},
    "3 piksel sağa kaydırılmış": {"shift_x": 3},
    "%80'e küçültülmüş": {"scale": 0.8},
    "%120'ye büyütülmüş": {"scale": 1.2},
}


@torch.no_grad()
def robustness_table(models, x, y, transforms=ROBUSTNESS_TRANSFORMS):
    """Her model için, validation görüntülerine aynı dönüşüm uygulandığında accuracy. {model adı: model}"""
    names = list(models)
    lines = ["| Validation görüntüleri | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for title, kwargs in transforms.items():
        xt = fixed_transform(x, **kwargs) if kwargs else x
        cells = [f"%{(predict(model, xt).argmax(dim=1) == y.cpu()).float().mean().item() * 100:.2f}"
                 for model in models.values()]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_probabilities(probabilities, labels, title=None):
    """Her rakam için modelin 10 sınıfa verdiği olasılıklar (satır: bizim rakamımız, sütun: sınıf)."""
    probabilities = np.asarray(probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(probabilities) + 1.2))
    ax.imshow(probabilities, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(probabilities):
        for j, value in enumerate(row):
            if value >= 0.01:
                ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.6 else plot_style.INK_SECONDARY)
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(probabilities)))
    ax.set_yticklabels([f"{i + 1}. rakam ({label})" for i, label in enumerate(labels)], fontsize=8)
    ax.set_xlabel("Sınıf (olasılık, %)")
    ax.grid(False)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def save_figures(out_dir, grays, digits, results_by_model, labels, mnist_examples):
    """Bulunan rakamlar, ön işlemeler, final modelin tahminleri ve olasılıkları, iki modelin tahminleri."""
    final_name, other_name = list(results_by_model)
    final, other = results_by_model[final_name], results_by_model[other_name]
    rows = [r["images"] for r in final.values()]

    def predictions(results):
        return list(zip(results["predicted"], results["confidence"]))

    figures = {
        "tespit.png": plot_detection(grays, digits),
        "on_isleme.png": plot_digits_table(rows + [mnist_examples], list(PIPELINES.values()) + ["MNIST örnekleri"],
                                           labels),
        "tahminler.png": plot_digits_table(rows, list(PIPELINES.values()), labels,
                                           predictions=[predictions(r) for r in final.values()],
                                           title="Final modelin tahminleri (yeşil doğru, kırmızı yanlış)"),
        "olasiliklar.png": plot_probabilities(final["mnist"]["probabilities"], labels,
                                              title="MNIST tarzı ön işleme: sınıf olasılıkları (%)"),
        "iki_model.png": plot_digits_table([final["mnist"]["images"]] * 2, [final_name, other_name], labels,
                                           predictions=[predictions(final["mnist"]), predictions(other["mnist"])],
                                           title="MNIST tarzı ön işleme: iki modelin tahminleri"),
    }
    for name, fig in figures.items():
        fig.savefig(Path(out_dir) / name)
        plt.close(fig)


def main(out_dir=RESULTS_DIR, processed_dir=PROCESSED_DIR):
    from augment import DEFAULT_AUGMENT
    from data import get_splits, load_data
    from device import get_device
    from final import RESULTS_DIR as FINAL_DIR, load_model, save_model, train_final

    student_number = sys.argv[1]
    plot_style.apply()
    device = get_device()
    model, (mean, std) = load_model(FINAL_DIR / "model.pt")
    digits, grays = extract_digits(list_photos())
    print(f"{len(digits)} rakam bulundu, {len(student_number)} bekleniyordu")
    if len(digits) != len(student_number):
        raise SystemExit("Rakam sayısı numarayla uyuşmuyor; fotoğrafları ve numarayı kontrol edin.")

    # Karşılaştırma modeli: final ayarın aynısı, yalnızca veri artırma tersine çevrilmiş (test seti yok)
    config = {k: (tuple(v) if k == "hidden" else v)
              for k, v in torch.load(FINAL_DIR / "model.pt", weights_only=True)["config"].items() if k != "seed"}
    has_augment = bool(config.get("augment"))
    final_name = "Final model (veri artırmalı)" if has_augment else "Final model (veri artırmasız)"
    other_name = "Veri artırmasız model" if has_augment else "Veri artırmalı model"
    data, _ = load_data(device=device)
    other_runs = train_final({**config, "augment": None if has_augment else DEFAULT_AUGMENT}, data, device)
    other_best = max(other_runs, key=lambda r: r["history"][r["best_epoch"] - 1]["val_acc"])
    other = other_best["model"].to(device).eval()
    robustness = robustness_table({final_name: model.to(device), other_name: other}, *data["val"])
    model.cpu()
    other.cpu()

    results = {final_name: evaluate_pipelines(model, mean, std, digits, student_number),
               other_name: evaluate_pipelines(other, mean, std, digits, student_number)}
    print(summary_table(results) + "\n\n" + robustness)
    save_processed(results[final_name]["mnist"]["images"], student_number, processed_dir)
    out = save_results(results, student_number, out_dir, robustness)
    save_model(other_best, (mean, std), out / "model_karsilastirma.pt")
    mnist_examples = list(get_splits()["train"][0][:len(digits)].numpy())
    save_figures(out, grays, digits, results, list(student_number), mnist_examples)
    print("Kaydedildi:", out)


if __name__ == "__main__":
    main()
