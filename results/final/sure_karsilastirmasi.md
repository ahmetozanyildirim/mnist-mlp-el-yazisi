| Ortam | PyTorch | MKL | İş parçacığı | İşlemci | Epoch süresi | Hız (en yavaşa göre) |
|---|---|---|---|---|---|---|
| GPU (ROCm) | 2.14.0+rocm10.1.0 | yok | 8 | AMD Radeon RX 9070 XT | 0.09 sn | 31.0× |
| CPU, OpenBLAS (ROCm PyTorch) | 2.14.0+rocm10.1.0 | yok | 8 | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD | 2.80 sn | 1.0× |
| CPU, MKL (CPU PyTorch) | 2.11.0+cpu | var | 8 | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD | 0.16 sn | 17.4× |
