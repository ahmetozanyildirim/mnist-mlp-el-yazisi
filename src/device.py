"""Eğitimde kullanılacak cihazı (GPU veya CPU) seçer."""
import torch


def get_device():
    """En güçlü GPU'yu seçer; GPU yoksa CPU döner.

    Bu bilgisayarda PyTorch iki GPU görüyor: işlemcinin içindeki dahili grafik birimi
    (gfx1036) ve RX 9070 XT (gfx1201). Dahili GPU için ROCm kütüphaneleri kurulu
    olmadığından onunla işlem yapmak programı çökertir. Bu yüzden cihazı hesaplama
    birimi (CU) sayısına göre seçiyoruz. Bellek miktarı iyi bir ölçüt değil, çünkü
    dahili GPU sistem RAM'ini paylaştığı için büyük görünür.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.get_device_properties(i).multi_processor_count,
    )
    # Varsayılan GPU'yu da ayarla ki .cuda() / synchronize() gibi çağrılar yanlış cihaza gitmesin
    torch.cuda.set_device(best)
    return torch.device(f"cuda:{best}")
