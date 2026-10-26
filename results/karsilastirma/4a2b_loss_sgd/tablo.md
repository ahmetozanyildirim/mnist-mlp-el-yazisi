| Seçenek | Val acc | Val loss (her biri kendi loss'u, kıyaslanamaz) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| Cross-entropy | %98.03 ± 0.08 | 0.0713 ± 0.0024 | %93.67 ± 1.54 | %99.90 ± 0.03 | 9.3 ± 0.6 | 15.0 ± 0.1 sn |
| MSE (softmax + one-hot) | %95.62 ± 0.03 | 0.0068 ± 0.0001 | %75.03 ± 2.38 | %96.71 ± 0.05 | 20.0 ± 0.0 | 15.9 ± 0.2 sn |
| Hinge (multi-margin) | %97.86 ± 0.08 | 0.0088 ± 0.0003 | %92.96 ± 0.12 | %99.65 ± 0.01 | 20.0 ± 0.0 | 14.3 ± 0.1 sn |
