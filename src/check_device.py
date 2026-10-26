"""Ortam kontrolü: PyTorch sürümü, GPU görünüyor mu, GPU doğru hesaplıyor mu?

Çalıştırma:  python src/check_device.py
"""
import time

import torch

from device import get_device


def main():
    print(f"PyTorch sürümü  : {torch.__version__}")
    # ROCm sürümünde torch.version.hip dolu, torch.version.cuda boş olur
    print(f"HIP (ROCm)      : {torch.version.hip}")
    print(f"CUDA            : {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("GPU bulunamadı -> eğitim CPU'da yapılacak.")
        return

    # ROCm'lu PyTorch, AMD GPU'ları da "cuda" adıyla gösterir
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}           : {p.name} ({p.gcnArchName}, {p.multi_processor_count} CU)")

    device = get_device()
    props = torch.cuda.get_device_properties(device)
    print(f"Seçilen cihaz   : {device} -> {props.name}")
    print(f"GPU belleği     : {props.total_memory / 1024**3:.1f} GB")

    # Doğruluk testi: aynı matris çarpımını CPU ve GPU'da yapıp sonuçları karşılaştır
    torch.manual_seed(0)
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    cpu_result = a @ b
    gpu_result = (a.to(device) @ b.to(device)).cpu()
    max_diff = (cpu_result - gpu_result).abs().max().item()
    print(f"CPU-GPU farkı   : {max_diff:.2e} (küçükse GPU doğru hesaplıyor)")

    # Zaman ölçümü: GPU asenkron çalışır, synchronize() olmadan ölçülen süre yanıltıcıdır
    x = torch.randn(4096, 4096, device=device)
    _ = x @ x  # ısınma: ilk çağrıda kernel yükleme maliyeti var
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    _ = x @ x
    no_sync = time.perf_counter() - start
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    _ = x @ x
    torch.cuda.synchronize(device)
    with_sync = time.perf_counter() - start

    print(f"synchronize yok : {no_sync * 1000:.2f} ms (yanlış: sadece işi kuyruğa koyma süresi)")
    print(f"synchronize var : {with_sync * 1000:.2f} ms (doğru: işin gerçek süresi)")


if __name__ == "__main__":
    main()
