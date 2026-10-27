- **TPE (Optuna) #26:** 2 × 1024, Adam, lr 0.00022, batch 64, dropout 0.30, wd 1e-06, BN, veri artırma hafif
- **TPE (Optuna) #22:** 2 × 1024, mSGD (m = 0.94), lr 0.03, batch 64, dropout 0.38, wd 2e-05, BN, veri artırma hafif
- **TPE (Optuna) #28:** 3 × 1024, Adam, lr 0.00021, batch 64, dropout 0.26, wd 2e-05, BN, veri artırma hafif
- **TPE (Optuna) #27:** 3 × 1024, Adam, lr 0.00021, batch 64, dropout 0.22, wd 5e-05, BN, veri artırma hafif
- **TPE (Optuna) #23:** 2 × 1024, mSGD (m = 0.94), lr 0.026, batch 64, dropout 0.30, wd 3e-05, BN, veri artırma hafif
- **4A sonucu:** 2 × 256, mSGD (m = 0.90), lr 0.08, batch 512, dropout 0.00, wd 0e+00, veri artırma hafif

| Seçenek | Parametre | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (100 epoch) |
|---|---|---|---|---|---|---|---|
| TPE (Optuna) #26 | 1 867 786 | %99.15 ± 0.04 | 0.0275 ± 0.0010 | %96.53 ± 0.16 | %99.43 ± 0.11 | 27.0 ± 5.0 | 43.1 ± 6.7 sn |
| TPE (Optuna) #22 | 1 867 786 | %99.14 ± 0.06 | 0.0269 ± 0.0012 | %96.01 ± 0.29 | %99.45 ± 0.05 | 28.7 ± 4.9 | 43.0 ± 8.0 sn |
| TPE (Optuna) #28 | 2 919 434 | %99.00 ± 0.08 | 0.0293 ± 0.0023 | %96.63 ± 0.27 | %99.24 ± 0.15 | 19.3 ± 3.8 | 38.2 ± 5.8 sn |
| TPE (Optuna) #27 | 2 919 434 | %99.08 ± 0.11 | 0.0282 ± 0.0022 | %96.69 ± 0.14 | %99.33 ± 0.18 | 21.7 ± 6.5 | 42.0 ± 10.1 sn |
| TPE (Optuna) #23 | 1 867 786 | %99.18 ± 0.07 | 0.0249 ± 0.0025 | %96.14 ± 0.26 | %99.53 ± 0.10 | 35.7 ± 11.2 | 50.4 ± 14.2 sn |
| 4A sonucu | 269 322 | %98.89 ± 0.16 | 0.0357 ± 0.0047 | %93.93 ± 0.89 | %99.26 ± 0.30 | 24.0 ± 14.1 | 3.6 ± 1.8 sn |
