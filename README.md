# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

Yapay Sinir Ağları dersi ödevi. MNIST veri seti üzerinde sıfırdan eğitilen bir Multi-Layer Perceptron (MLP) ile el yazısı rakam tanıma; farklı model yapılandırmalarının karşılaştırılması ve modelin kendi el yazımla yazdığım öğrenci numarası üzerindeki başarısının değerlendirilmesi.

> Bu dosya ödevin raporudur. Bölümler deneyler ilerledikçe doldurulmaktadır.

## Ödev Gereksinimleri ve Karşılandığı Yer

| Gereksinim | Durum | Nerede |
|---|---|---|
| Model MLP (CNN yok), hazır model yok, sıfırdan eğitim | ✅ | [Bölüm 4.1](#41-model) |
| En az 2 gizli katman, ReLU aktivasyonu | ✅ | [Bölüm 4.1](#41-model) |
| Uygun sınıflandırma loss'u, SGD veya Adam | ✅ cross-entropy + Adam / ⏳ alternatiflerle karşılaştırma | [Bölüm 4.2](#42-hiperparametreler), [Bölüm 5](#5-karşılaştırmalı-deneyler) |
| Train / validation / test ayrımı açıkça gösterilmiş | ✅ | [Bölüm 3](#3-veri-seti-ve-trainvalidationtest-ayrımı) |
| Test verisi model geliştirmede kullanılmamış | ✅ kodla garanti altında | [Bölüm 3](#3-veri-seti-ve-trainvalidationtest-ayrımı) |
| Deney 1: train/val loss, train/val accuracy, test accuracy, eğitim süresi | ✅ | [Bölüm 4.4](#44-sonuçlar) |
| Deney 1: loss ve accuracy grafikleri | ✅ | [Bölüm 4.4](#44-sonuçlar) |
| Model yapılandırmalarının karşılaştırılması | ⏳ | [Bölüm 5](#5-karşılaştırmalı-deneyler), [Bölüm 6](#6-ortak-hiperparametre-araması) |
| Kullanılan hiperparametreler raporda | ✅ Deney 1 / ⏳ diğer deneyler | [Bölüm 4.2](#42-hiperparametreler), Bölüm 5–7 |
| El yazısı görüntüleri: gri, 28×28, MNIST renk uyumu, normalizasyon | ⏳ | [Bölüm 8](#8-kendi-el-yazım-ile-değerlendirme) |
| El yazısıyla öğrenci numarası görselleri repoda | ⏳ | [`el_yazisi/`](el_yazisi/) |
| Deney sonuçları hakkında genel değerlendirme | ⏳ | [Bölüm 9](#9-genel-değerlendirme) |
| Kodun çalıştırılabilir olması | ✅ kurulum, veri, Deney 1 / ⏳ diğer deneyler | [Bölüm 1](#1-ortam-ve-kurulum) |

## İçindekiler
1. [Ortam ve Kurulum](#1-ortam-ve-kurulum)
2. [Proje Yapısı](#2-proje-yapısı)
3. [Veri Seti ve Train/Validation/Test Ayrımı](#3-veri-seti-ve-trainvalidationtest-ayrımı)
4. [Deney 1 – Temel MLP Modeli](#4-deney-1--temel-mlp-modeli)
5. [Karşılaştırmalı Deneyler](#5-karşılaştırmalı-deneyler)
6. [Ortak Hiperparametre Araması](#6-ortak-hiperparametre-araması)
7. [Final Model ve Test Sonucu](#7-final-model-ve-test-sonucu)
8. [Kendi El Yazım ile Değerlendirme](#8-kendi-el-yazım-ile-değerlendirme)
9. [Genel Değerlendirme](#9-genel-değerlendirme)

## 1. Ortam ve Kurulum

| | |
|---|---|
| İşletim sistemi | Windows 11 |
| CPU | AMD Ryzen 7 9800X3D |
| GPU | AMD Radeon RX 9070 XT (ROCm) |
| Python | 3.12 |
| Kütüphane | PyTorch (ROCm sürümü) |

Kurulum:

```bash
python -m venv .venv
.venv\Scripts\activate

# Yalnızca AMD GPU (RX 9070 serisi) için: önce ROCm'lu PyTorch kurulur
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0" "torchvision[device-gfx1201]==0.29.0a0+rocm10.1.0"

# Diğer bağımlılıklar (CPU veya NVIDIA kullananlar doğrudan bu adımla başlayabilir)
python -m pip install -r requirements.txt

# Ortam kontrolü
python src/check_device.py
```

Kod GPU bulamazsa otomatik olarak CPU'da çalışır.

Deneyleri çalıştırma (komutlar bu klasörde çalıştırılır; sonuçlar `results/` altına yazılır):

| Komut | Ne yapar | Süre (RX 9070 XT) |
|---|---|---|
| `python src/explore_data.py` | Veri ayrımı, sınıf dağılımı ve örnek görsel grafikleri (Bölüm 3) | birkaç saniye |
| `python src/deney1.py` | Deney 1: baseline MLP, 3 seed × 20 epoch, test değerlendirmesi (Bölüm 4) | ~1 dakika |

## 2. Proje Yapısı

```
├── README.md            ← rapor
├── requirements.txt
├── src/                 ← kaynak kodlar
├── el_yazisi/
│   ├── ham/             ← el yazısıyla öğrenci numarası fotoğrafları (ham hâli)
│   └── islenmis/        ← MNIST formatına getirilmiş 28×28 hâli
└── results/             ← deney sonuçları ve grafikler
```

## 3. Veri Seti ve Train/Validation/Test Ayrımı

Kod: [`src/data.py`](src/data.py), analiz: [`src/explore_data.py`](src/explore_data.py)

MNIST; 28×28 piksel, gri seviyeli (0 = siyah zemin, 255 = beyaz mürekkep), 10 sınıflı el yazısı rakam görüntülerinden oluşur.

![MNIST örnekleri](results/data_samples.png)

### Ayrım

| Parça | Kaynak | Boyut | Kullanım amacı |
|---|---|---|---|
| Train | MNIST eğitim kısmı | 50.000 | Ağırlıkların öğrenilmesi |
| Validation | MNIST eğitim kısmı | 10.000 | Hiperparametre ve model seçimi, overfitting takibi |
| Test | MNIST test kısmı | 10.000 | Yalnızca en sonda, bir kez: nihai performans |

- **Neden 50.000 / 10.000?** Validation seti test setiyle aynı boyutta olduğu için doğruluk tahmini aynı hassasiyette (bir örnek = %0,01) olur. Daha küçük bir validation seti (ör. 6.000) train'e 4.000 örnek kazandırırdı, ancak model seçiminde ölçüm gürültüsü artardı.
- **Neden rastgele ayrım (stratified değil)?** 10.000 örneklik rastgele bir parçada sınıf oranları zaten dengeli kalır (aşağıdaki grafik). Stratified ayrım oranları birebir eşitlerdi ama bu veri boyutunda pratik bir fark yaratmaz.
- **Tekrarlanabilirlik:** Karıştırma sabit bir seed (`42`) ile yapılır; her çalıştırmada aynı 10.000 görüntü validation'a düşer.
- **Test setinin korunması:** `load_data()` test setini varsayılan olarak döndürmez (`include_test=False`). Test verisi yalnızca açıkça istendiğinde yüklenir; böylece model geliştirme sırasında yanlışlıkla kullanılamaz.

![Sınıf dağılımı](results/class_distribution.png)

| Rakam | Train | Validation | Test |
|---|---|---|---|
| 0 | 4930 (%9.9) | 993 (%9.9) | 980 (%9.8) |
| 1 | 5668 (%11.3) | 1074 (%10.7) | 1135 (%11.4) |
| 2 | 4942 (%9.9) | 1016 (%10.2) | 1032 (%10.3) |
| 3 | 5089 (%10.2) | 1042 (%10.4) | 1010 (%10.1) |
| 4 | 4860 (%9.7) | 982 (%9.8) | 982 (%9.8) |
| 5 | 4507 (%9.0) | 914 (%9.1) | 892 (%8.9) |
| 6 | 4923 (%9.8) | 995 (%9.9) | 958 (%9.6) |
| 7 | 5254 (%10.5) | 1011 (%10.1) | 1028 (%10.3) |
| 8 | 4846 (%9.7) | 1005 (%10.1) | 974 (%9.7) |
| 9 | 4981 (%10.0) | 968 (%9.7) | 1009 (%10.1) |

Sınıflar arasında en büyük fark 1 (%11,3) ile 5 (%9,0) arasındadır. Bu dengesizlik küçük olduğu için sınıf ağırlıklandırması gibi ek bir önleme gerek görülmemiştir; accuracy anlamlı bir metrik olarak kullanılabilir.

### Normalizasyon istatistikleri

Her görüntü 784 uzunluğunda bir vektöre düzleştirilir (MLP girdisi). Standardizasyon `(x − ortalama) / std` ile yapılır ve **ortalama ile std yalnızca train parçasından hesaplanır**:

| Hesaplandığı veri | Ortalama | Std |
|---|---|---|
| Yalnızca train (kullanılan) | 0,1306 | 0,3081 |
| Train + validation + test | 0,1309 | 0,3084 |

Validation ve test verisi istatistik hesabına katılsaydı, model değerlendirme verisi hakkında önceden bilgi almış olurdu (**veri sızıntısı**). MNIST'te iki değer arasındaki fark çok küçüktür. Yine de doğru yöntem, gerçek hayattaki durumu taklit etmektir: model eğitilirken gelecekteki verinin istatistiği bilinmez. Aynı train istatistikleri kendi el yazısı görüntülerimize de uygulanacaktır.

Ortalamanın düşük (0,13) olmasının nedeni piksellerin **%80,9'unun tamamen siyah** olmasıdır; rakam görüntünün yalnızca küçük bir bölümünü kaplar.

Alternatif normalizasyon yöntemleri (ham [0–255], [0–1] ölçekleme) Bölüm 5'te karşılaştırılacaktır.

## 4. Deney 1 – Temel MLP Modeli

Kod: [`src/deney1.py`](src/deney1.py), [`src/model.py`](src/model.py), [`src/train.py`](src/train.py) · Çalıştırma: `python src/deney1.py` (GPU'da ~1 dakika) · Ham sonuçlar: [`results/deney1/sonuclar.json`](results/deney1/sonuclar.json)

### 4.1 Model

```
girdi: 784 piksel (28×28 görüntü düzleştirilmiş, standardize edilmiş)
  ├─ Linear 784 → 256  +  ReLU     1. gizli katman
  ├─ Linear 256 → 128  +  ReLU     2. gizli katman
  └─ Linear 128 → 10               çıktı: her rakam için bir skor (logit)
```

- Yalnızca tam bağlantılı (`nn.Linear`) katmanlardan oluşan bir MLP. CNN katmanı yok.
- Toplam **235.146** parametre; bunların %85'i ilk katmanda.
- Model **sıfırdan** eğitildi. Ağırlıklar PyTorch'un varsayılan rastgele başlangıç değerleriyle başlıyor; hazır veya önceden eğitilmiş ağırlık kullanılmadı.
- Softmax modelin içinde değil, loss fonksiyonunun (`nn.CrossEntropyLoss`) içinde uygulanıyor.

### 4.2 Hiperparametreler

| Hiperparametre | Değer | Not |
|---|---|---|
| Gizli katmanlar | 2 katman: 256 → 128 nöron | Ödev şartı: en az 2 gizli katman |
| Aktivasyon | ReLU | Ödev şartı |
| Loss | Cross-entropy | Sınıflandırma için standart loss |
| Optimizer | Adam | Ödev şartı: SGD veya Adam |
| Learning rate | 0.001 | Adam'ın önerilen varsayılan değeri (Kingma & Ba, 2015) |
| Batch boyutu | 64 | Epoch başına 782 adım |
| Epoch sayısı | 20 | Early stopping yok |
| Regularization | Yok | Dropout 0, weight decay 0, batch normalization yok |
| Girdi normalizasyonu | Standardizasyon | Train istatistikleriyle: ortalama 0.1306, std 0.3081 (Bölüm 3) |
| Başlangıç ağırlıkları | PyTorch varsayılanı (`nn.Linear`) | |
| Model seçimi | En düşük validation loss'lu epoch | Eğitim sonunda o epoch'un ağırlıkları geri yüklenir |
| Tekrar | 3 seed (0, 1, 2) | Sonuçlar ortalama ± standart sapma olarak verilir |

Bu değerler bilinçli olarak sade ve yaygın başlangıç değerleri olarak seçildi. Her birinin alternatifleri Bölüm 5'te deneyle karşılaştırılacak.

### 4.3 Ölçüm yöntemi

- **Train ve validation metrikleri:** Her epoch sonunda, train ve validation setlerinin **tamamında**, model eval modundayken ölçüldü. Epoch boyunca batch'lerin ortalaması kullanılmadı; o yöntem, epoch boyunca değişen bir modeli ölçtüğü için validation ile adil kıyaslanmaz.
- **Raporlanan model:** Her seed için validation loss'un en düşük olduğu epoch'un ağırlıkları. Tablodaki train, validation ve test değerleri bu modele aittir.
- **Test seti:** Yalnızca bu tabloyu raporlamak için bir kez kullanıldı. Hiçbir hiperparametre kararı test sonucuna bakılarak verilmedi.
- **Eğitim süresi:**
  - Yalnızca optimizasyon adımlarını kapsar; epoch sonu ölçümler dahil değildir.
  - GPU asenkron çalıştığı için süre `torch.cuda.synchronize()` ile ölçüldü.
  - İlk GPU çağrısındaki tek seferlik kütüphane yükleme maliyeti, ölçümden önce yapılan bir ısınma turuyla ölçüm dışında bırakıldı.
- **Tekrarlanabilirlik:** Seed'ler sabit. Ancak kullanılan ROCm GPU'da aynı seed bile program yeniden başlatıldığında son basamaklarda farklı sonuç verebiliyor. Kod tekrar çalıştırıldığında sayılar küçük farklarla değişebilir; bu yüzden sonuçlar 3 seed'in ortalaması ± standart sapması olarak verildi.

### 4.4 Sonuçlar

Donanım: AMD Radeon RX 9070 XT (ROCm), PyTorch 2.14.0+rocm10.1.0, Python 3.12

| Seed | En iyi epoch | Train loss | Val loss | Train acc | Val acc | Test loss | Test acc | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|---|---|
| 0 | 9 | 0.0136 | 0.0847 | %99.57 | %97.94 | 0.0895 | %97.82 | 16.3 sn |
| 1 | 4 | 0.0314 | 0.0765 | %99.09 | %97.59 | 0.0747 | %97.76 | 16.6 sn |
| 2 | 7 | 0.0150 | 0.0747 | %99.56 | %97.96 | 0.0749 | %97.92 | 16.5 sn |
| **Ortalama ± std** | 6.7 ± 2.5 | 0.0200 ± 0.0099 | 0.0786 ± 0.0053 | %99.41 ± 0.27 | %97.83 ± 0.21 | 0.0797 ± 0.0084 | %97.83 ± 0.08 | 16.5 ± 0.1 sn |

**Özetle:** Training loss **0.020**, validation loss **0.079**, training accuracy **%99.41**, validation accuracy **%97.83**, test accuracy **%97.83**. Eğitim süresi 20 epoch için **16.5 sn** (epoch başına ~0.82 sn).

![Deney 1 loss ve accuracy eğrileri](results/deney1/egriler.png)

*Çizgiler 3 seed'in ortalaması, gölgeler ± standart sapma. Kesikli dikey çizgiler her seed'in en düşük validation loss'lu epoch'u.*

### 4.5 Değerlendirme

1. **Validation, testi iyi temsil ediyor.** Test doğruluğu (%97.83 ± 0.08) validation doğruluğuyla (%97.83 ± 0.21) aynı. Bu yüzden sonraki deneylerde kararları validation'a bakarak vermek güvenilir.
2. **Overfitting var.**
   - Ortalama validation loss en düşük değerine 5. epoch'ta ulaşıyor (0.087) ve 20. epoch'ta 0.136'ya çıkıyor (%56 artış).
   - Aynı sürede training loss 0.109'dan 0.013'e düşüyor.
   - 20. epoch'ta training accuracy %99.60, validation accuracy %97.72.

   Model belli bir noktadan sonra genel kalıpları öğrenmek yerine train setine özgü ayrıntıları ezberliyor. Baseline'da hiçbir regularization olmadığı için bu beklenen bir sonuç.
3. **Validation loss artarken validation accuracy düşmüyor.** Validation accuracy 5. epoch'ta %97.52, 20. epoch'ta %97.72. Cross-entropy, modelin yanlış tahminlerine ne kadar emin olduğunu da cezalandırır. Eğitim ilerledikçe model aşırı özgüvenli hâle geliyor: yanlış bildiği görüntü sayısı pek değişmiyor, ama bu yanlışlarda giderek daha emin oluyor. Accuracy bunu görmüyor, loss görüyor.
4. **En iyi epoch seed'e göre 4 ile 9 arasında değişiyor**, çünkü ortalama validation loss bu aralıkta oldukça düz seyrediyor. Model seçme ölçütü (validation loss mu, validation accuracy mi) ve early stopping Bölüm 5'te incelenecek.
5. **Seed'ler arası değişkenlik küçük.** Accuracy'deki standart sapma 0.1–0.3 puan. Sonraki karşılaştırmalarda bu büyüklükteki farklar anlamlı kabul edilmeyecek.
6. **Eğitim süresi.** 20 epoch GPU'da 16.5 sn sürdü. Validation loss 5. epoch civarında en düşük seviyeye indiği için 20 epoch'un büyük kısmı genellemeye katkı sağlamadı. Early stopping hem overfitting'i hem eğitim süresini azaltabilir (Bölüm 5).

## 5. Karşılaştırmalı Deneyler
_(Aşama 4A'da doldurulacak)_

## 6. Ortak Hiperparametre Araması
_(Aşama 4B–4C'de doldurulacak)_

## 7. Final Model ve Test Sonucu
_(Aşama 5'te doldurulacak)_

## 8. Kendi El Yazım ile Değerlendirme
_(Aşama 6'da doldurulacak)_

## 9. Genel Değerlendirme
_(Aşama 7'de doldurulacak)_
