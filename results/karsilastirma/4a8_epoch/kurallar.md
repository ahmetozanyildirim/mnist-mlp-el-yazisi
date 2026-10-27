| Kural | Durduğu epoch | Seçilen epoch | Val acc (seçim yarısı) | Holdout acc | Holdout loss | Eğitim süresi |
|---|---|---|---|---|---|---|
| 10 epoch, en düşük val loss | 10.0 ± 0.0 | 8.7 ± 0.6 | %98.45 ± 0.14 | %98.35 ± 0.04 | 0.0483 ± 0.0020 | 1.4 ± 0.1 sn |
| 20 epoch, en düşük val loss (mevcut) | 20.0 ± 0.0 | 17.7 ± 1.2 | %98.81 ± 0.12 | %99.02 ± 0.02 | 0.0348 ± 0.0009 | 2.8 ± 0.1 sn |
| 40 epoch, en düşük val loss | 40.0 ± 0.0 | 39.3 ± 1.2 | %99.15 ± 0.10 | %99.05 ± 0.06 | 0.0317 ± 0.0009 | 5.4 ± 0.2 sn |
| 100 epoch, en düşük val loss | 100.0 ± 0.0 | 60.7 ± 22.1 | %99.21 ± 0.07 | %99.12 ± 0.07 | 0.0318 ± 0.0014 | 13.4 ± 0.2 sn |
| 20 epoch, en yüksek val acc | 20.0 ± 0.0 | 19.7 ± 0.6 | %98.93 ± 0.03 | %98.87 ± 0.13 | 0.0362 ± 0.0027 | 2.8 ± 0.1 sn |
| 100 epoch, en yüksek val acc | 100.0 ± 0.0 | 83.0 ± 15.1 | %99.29 ± 0.04 | %99.13 ± 0.11 | 0.0310 ± 0.0052 | 13.4 ± 0.2 sn |
| Early stopping: val loss, patience 5 | 32.0 ± 11.4 | 27.0 ± 11.4 | %99.03 ± 0.15 | %99.04 ± 0.07 | 0.0333 ± 0.0016 | 4.4 ± 1.5 sn |
| Early stopping: val loss, patience 10 | 39.7 ± 9.3 | 29.7 ± 9.3 | %99.07 ± 0.12 | %98.96 ± 0.17 | 0.0347 ± 0.0034 | 5.4 ± 1.3 sn |
| Early stopping: val acc, patience 5 | 31.3 ± 11.8 | 26.3 ± 11.8 | %99.03 ± 0.15 | %98.98 ± 0.16 | 0.0336 ± 0.0021 | 4.3 ± 1.6 sn |
| Early stopping: val acc, patience 10 | 47.7 ± 9.7 | 37.7 ± 9.7 | %99.15 ± 0.09 | %98.95 ± 0.17 | 0.0328 ± 0.0049 | 6.5 ± 1.1 sn |

```
100 epoch, en yüksek val acc             holdout acc %99.13 | en iyiden fark 0.00 puan -> en iyi               |  13.4 sn
100 epoch, en düşük val loss             holdout acc %99.12 | en iyiden fark 0.01 puan -> fark anlamlı DEĞİL   |  13.4 sn
40 epoch, en düşük val loss              holdout acc %99.05 | en iyiden fark 0.08 puan -> fark anlamlı DEĞİL   |   5.4 sn
Early stopping: val loss, patience 5     holdout acc %99.04 | en iyiden fark 0.09 puan -> fark anlamlı DEĞİL   |   4.4 sn  <- SEÇİLEN
20 epoch, en düşük val loss (mevcut)     holdout acc %99.02 | en iyiden fark 0.11 puan -> fark anlamlı DEĞİL   |   2.8 sn
Early stopping: val acc, patience 5      holdout acc %98.98 | en iyiden fark 0.15 puan -> fark anlamlı DEĞİL   |   4.3 sn
Early stopping: val loss, patience 10    holdout acc %98.96 | en iyiden fark 0.17 puan -> fark anlamlı DEĞİL   |   5.4 sn
Early stopping: val acc, patience 10     holdout acc %98.95 | en iyiden fark 0.18 puan -> fark anlamlı DEĞİL   |   6.5 sn
20 epoch, en yüksek val acc              holdout acc %98.87 | en iyiden fark 0.26 puan -> fark anlamlı DEĞİL   |   2.8 sn
10 epoch, en düşük val loss              holdout acc %98.35 | en iyiden fark 0.78 puan -> anlamlı şekilde kötü |   1.4 sn
```
