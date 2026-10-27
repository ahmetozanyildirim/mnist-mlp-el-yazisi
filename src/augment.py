"""Veri artırma (data augmentation): eğitim sırasında her batch'teki görüntüleri rastgele döndürür, kaydırır, ölçekler.

Model her epoch aynı rakamın biraz farklı bir hâlini görür. Amaç, gerçek el yazısında görülen eğiklik, kayma ve
boyut farklarına karşı modeli daha sağlam yapmak (Aşama 6). Görüntüler GPU'da, batch hâlinde dönüştürülür.
"""
import math

import torch
import torch.nn.functional as F

# Her dönüşümün en fazla ne kadar olabileceği (iki yönde de): derece, piksel, oran
AUGMENT_LEVELS = {
    "hafif": {"rotation": 10, "shift": 2, "scale": 0.10},
    "orta": {"rotation": 15, "shift": 3, "scale": 0.15},
    "guclu": {"rotation": 20, "shift": 4, "scale": 0.20},
}
DEFAULT_AUGMENT = AUGMENT_LEVELS["orta"]


def affine_matrices(angle_deg, shift_x, shift_y, scale):
    """Döndürme (derece), kaydırma (piksel) ve ölçekleme için affine_grid'in beklediği 2×3 matrisler.

    Argümanlar aynı uzunlukta tensörlerdir (her görüntü için bir değer).
    """
    angle = angle_deg * math.pi / 180
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    # affine_grid'de koordinatlar [-1, 1] aralığında: görüntünün yarı genişliği (14 piksel) = 1
    tx, ty = -shift_x / 14, -shift_y / 14
    return torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)


def transform_images(x, matrices):
    """(N, 784) görüntüleri verilen dönüşümlerle değiştirir."""
    images = x.reshape(-1, 1, 28, 28)
    grid = F.affine_grid(matrices, images.shape, align_corners=False)
    # border: görüntünün dışına düşen yerler kenar pikseliyle, yani arka planla doldurulur
    return F.grid_sample(images, grid, padding_mode="border", align_corners=False).reshape(len(x), -1)


def augment_batch(x, augment, generator=None):
    """Her görüntüye ayrı, rastgele bir döndürme + kaydırma + ölçekleme uygular."""
    def uniform(limit):
        return (torch.rand(len(x), device=x.device, generator=generator) * 2 - 1) * limit

    matrices = affine_matrices(uniform(augment["rotation"]), uniform(augment["shift"]), uniform(augment["shift"]),
                               1 + uniform(augment["scale"]))
    return transform_images(x, matrices)


def fixed_transform(x, rotation=0.0, shift_x=0.0, shift_y=0.0, scale=1.0):
    """Bütün görüntülere aynı dönüşümü uygular (ör. validation setini 20° döndürüp modeli sınamak için)."""
    def full(value):
        return torch.full((len(x),), float(value), device=x.device)

    return transform_images(x, affine_matrices(full(rotation), full(shift_x), full(shift_y), full(scale)))
