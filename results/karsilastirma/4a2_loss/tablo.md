| Seçenek | Val acc | Val loss (her biri kendi loss'u, kıyaslanamaz) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| Cross-entropy | %97.68 ± 0.10 | 0.0782 ± 0.0027 | %95.93 ± 0.30 | %99.21 ± 0.24 | 5.7 ± 2.1 | 17.6 ± 0.2 sn |
| MSE (softmax + one-hot) | %97.89 ± 0.08 | 0.0034 ± 0.0001 | %95.37 ± 0.35 | %99.17 ± 0.22 | 11.3 ± 3.2 | 18.2 ± 0.2 sn |
| Hinge (multi-margin) | %97.45 ± 0.12 | 0.0098 ± 0.0004 | %95.07 ± 0.63 | %98.84 ± 0.32 | 5.3 ± 1.5 | 16.7 ± 0.2 sn |
