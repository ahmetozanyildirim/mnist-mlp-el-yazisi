| Yöntem | En iyi val acc | İlk 10 denemenin ortalaması | Son 10 denemenin ortalaması | Öğrenemeyen deneme | Toplam süre |
|---|---|---|---|---|---|
| Rastgele arama | %99.08 | %96.70 | %97.95 | 1 | 5.4 dk |
| TPE (Optuna) | %99.29 | %96.70 | %98.99 | 0 | 16.2 dk |

En iyi 10 deneme:

| # | Yöntem | Deneme | Ayar | Parametre | Val acc (tek seed) | En iyi epoch / eğitilen | Süre |
|---|---|---|---|---|---|---|---|
| 1 | TPE (Optuna) | 26 | 2 × 1024, Adam, lr 0.00022, batch 64, dropout 0.30, wd 1e-06, BN, veri artırma hafif | 1 867 786 | %99.29 | 42 / 47 | 72.0 sn |
| 2 | TPE (Optuna) | 22 | 2 × 1024, mSGD (m = 0.94), lr 0.03, batch 64, dropout 0.38, wd 2e-05, BN, veri artırma hafif | 1 867 786 | %99.26 | 28 / 33 | 43.3 sn |
| 3 | TPE (Optuna) | 28 | 3 × 1024, Adam, lr 0.00021, batch 64, dropout 0.26, wd 2e-05, BN, veri artırma hafif | 2 919 434 | %99.22 | 38 / 43 | 73.5 sn |
| 4 | TPE (Optuna) | 27 | 3 × 1024, Adam, lr 0.00021, batch 64, dropout 0.22, wd 5e-05, BN, veri artırma hafif | 2 919 434 | %99.17 | 22 / 27 | 46.9 sn |
| 5 | TPE (Optuna) | 23 | 2 × 1024, mSGD (m = 0.94), lr 0.026, batch 64, dropout 0.30, wd 3e-05, BN, veri artırma hafif | 1 867 786 | %99.16 | 22 / 27 | 36.2 sn |
| 6 | TPE (Optuna) | 29 | 4 × 1024, Adam, lr 0.00021, batch 64, dropout 0.23, wd 2e-06, BN, veri artırma hafif | 3 971 082 | %99.16 | 30 / 35 | 59.3 sn |
| 7 | TPE (Optuna) | 20 | 2 × 1024, mSGD (m = 0.94), lr 0.027, batch 64, dropout 0.44, wd 2e-06, BN, veri artırma hafif | 1 867 786 | %99.11 | 22 / 27 | 33.7 sn |
| 8 | Rastgele arama | 22 | 4 × 512, mSGD (m = 0.84), lr 0.045, batch 128, dropout 0.09, wd 6e-05, BN, veri artırma orta | 1 199 114 | %99.08 | 35 / 40 | 36.6 sn |
| 9 | TPE (Optuna) | 17 | 2 × 1024, mSGD (m = 0.94), lr 0.013, batch 64, dropout 0.35, wd 7e-06, BN, veri artırma orta | 1 867 786 | %99.07 | 21 / 26 | 35.1 sn |
| 10 | TPE (Optuna) | 18 | 2 × 1024, mSGD (m = 0.93), lr 0.034, batch 64, dropout 0.49, wd 1e-05, BN, veri artırma orta | 1 867 786 | %99.02 | 37 / 42 | 59.2 sn |
