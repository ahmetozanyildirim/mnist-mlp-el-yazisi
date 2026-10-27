# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

Yapay Sinir Ağları dersi ödevi. MNIST veri seti üzerinde sıfırdan eğitilen bir Multi-Layer Perceptron (MLP) ile el yazısı rakam tanıma; farklı model yapılandırmalarının karşılaştırılması ve modelin kendi el yazımla yazdığım öğrenci numarası üzerindeki başarısının değerlendirilmesi.

> Bu dosya ödevin raporudur. Bölümler deneyler ilerledikçe doldurulmaktadır.

## Ödev Gereksinimleri ve Karşılandığı Yer

| Gereksinim | Durum | Nerede |
|---|---|---|
| Model MLP (CNN yok), hazır model yok, sıfırdan eğitim | ✅ | [Bölüm 4.1](#41-model) |
| En az 2 gizli katman, ReLU aktivasyonu | ✅ | [Bölüm 4.1](#41-model), [Bölüm 5.4](#54-derinlik-ve-genişlik) |
| Uygun sınıflandırma loss'u, SGD veya Adam | ✅ cross-entropy (MSE ve hinge ile karşılaştırıldı) + momentumlu SGD (düz SGD ve Adam ile karşılaştırıldı) | [Bölüm 4.2](#42-hiperparametreler), [Bölüm 5.2](#52-loss-fonksiyonu), [Bölüm 5.3](#53-optimizer-ve-learning-rate) |
| Train / validation / test ayrımı açıkça gösterilmiş | ✅ | [Bölüm 3](#3-veri-seti-ve-trainvalidationtest-ayrımı) |
| Test verisi model geliştirmede kullanılmamış | ✅ kodla garanti altında | [Bölüm 3](#3-veri-seti-ve-trainvalidationtest-ayrımı) |
| Deney 1: train/val loss, train/val accuracy, test accuracy, eğitim süresi | ✅ | [Bölüm 4.4](#44-sonuçlar) |
| Deney 1: loss ve accuracy grafikleri | ✅ | [Bölüm 4.4](#44-sonuçlar) |
| Model yapılandırmalarının karşılaştırılması | ⏳ 8 deneyden 4'ü tamam (5.1 normalizasyon, 5.2 loss, 5.3 optimizer ve learning rate, 5.4 derinlik ve genişlik) | [Bölüm 5](#5-karşılaştırmalı-deneyler), [Bölüm 6](#6-ortak-hiperparametre-araması) |
| Kullanılan hiperparametreler raporda | ✅ Deney 1 / ⏳ diğer deneyler | [Bölüm 4.2](#42-hiperparametreler), Bölüm 5–7 |
| El yazısı görüntüleri: gri, 28×28, MNIST renk uyumu, normalizasyon | ⏳ | [Bölüm 8](#8-kendi-el-yazım-ile-değerlendirme) |
| El yazısıyla öğrenci numarası görselleri repoda | ⏳ | [`el_yazisi/`](el_yazisi/) |
| Deney sonuçları hakkında genel değerlendirme | ⏳ | [Bölüm 9](#9-genel-değerlendirme) |
| Kodun çalıştırılabilir olması | ✅ kurulum, veri, Deney 1, Bölüm 5.1–5.4 / ⏳ diğer deneyler | [Bölüm 1](#1-ortam-ve-kurulum) |

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
| `python src/karsilastirma.py 4a1_normalizasyon` | Normalizasyon karşılaştırması, Adam (Bölüm 5.1) | ~2.5 dakika |
| `python src/karsilastirma.py 4a1b_normalizasyon_sgd` | Normalizasyon karşılaştırması, ek analiz: SGD (Bölüm 5.1) | ~2.5 dakika |
| `python src/karsilastirma.py 4a2_loss` | Loss fonksiyonu karşılaştırması, Adam (Bölüm 5.2) | ~3 dakika |
| `python src/karsilastirma.py 4a2b_loss_sgd` | Loss fonksiyonu karşılaştırması, ek analiz: SGD (Bölüm 5.2) | ~2.5 dakika |
| `python src/karsilastirma.py 4a3_optimizer_lr` | Optimizer ve learning rate karşılaştırması: 3 optimizer × 4–5 learning rate (Bölüm 5.3) | ~12 dakika |
| `python src/karsilastirma.py 4a4_derinlik_genislik` | Derinlik ve genişlik: 1–4 gizli katman × 64–1024 nöron (Bölüm 5.4) | ~20 dakika |

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

Kod: [`src/karsilastirma.py`](src/karsilastirma.py) · Çalıştırma: `python src/karsilastirma.py <deney_adı>` · Ham sonuçlar: [`results/karsilastirma/`](results/karsilastirma/)

### 5.0 Yöntem

Deney 1'deki baseline'ın ayarları **tek tek** değiştirilerek karşılaştırıldı. Kurallar sonuçlar görülmeden önce belirlendi:

1. Her deneyde yalnızca **bir** hiperparametre değişir; geri kalan her şey sabit kalır.
2. Her seçenek 3 seed (0, 1, 2) ile 20 epoch eğitilir.
3. **Test seti kullanılmaz;** bu deneylerde yüklenmez bile. Bütün kararlar validation setine göre verilir.
4. **Birincil ölçüt:** en düşük validation loss'lu epoch'taki validation accuracy'nin 3 seed ortalaması. Bu, Deney 1'deki model seçimiyle aynı.
5. **Anlamlılık eşiği: 0.3 puan.** Deney 1'de aynı ayarlarla bile seed'ler arasındaki standart sapma ~0.2 puan çıktı. 3 seed'in ortalaması bu gürültüyü azaltır ama yok etmez; bu yüzden iki ortalama arasındaki 0.3 puandan küçük farklar anlamlı kabul edilmez.
6. Fark anlamlı değilse **ikincil ölçütlere** bakılır: yakınsama hızı (1. epoch validation accuracy), validation loss, eğitim süresi, sadelik ve sağlamlık. Bunlar da ayırt etmiyorsa mevcut ayarda kalınır.
7. Kazanan seçenek, bir sonraki deneyin başlangıç ayarına eklenir.

**Yöntemin sınırı:** Ayarları tek tek değiştirmek, ayarlar arasındaki etkileşimleri gözden kaçırabilir. Örneğin en iyi learning rate, batch boyutuna bağlı olabilir. Bu sınırlama Bölüm 6'da bütün ayarlar birlikte aranarak kontrol edilecek.

| # | Deney | Seçenekler | Karar |
|---|---|---|---|
| 5.1 | Girdi normalizasyonu | ham / min-max / standardizasyon | **Standardizasyon** |
| 5.2 | Loss fonksiyonu | cross-entropy / MSE / hinge | **Cross-entropy** |
| 5.3 | Optimizer ve learning rate | SGD / momentumlu SGD / Adam, her biri 4–5 learning rate ile | **Momentumlu SGD, lr = 0.01** |
| 5.4 | Derinlik ve genişlik | 1–4 gizli katman × 64–1024 nöron | **2 gizli katman × 256 nöron** |
| 5.5 | Aktivasyon | ReLU / Sigmoid / Tanh | ⏳ |
| 5.6 | Batch boyutu | 16 … 2048, learning rate sabit ya da batch ile orantılı | ⏳ |
| 5.7 | Regularization | dropout / weight decay / batch normalization / veri artırma | ⏳ |
| 5.8 | Epoch sayısı, early stopping ve model seçme ölçütü | | ⏳ |

### 5.1 Girdi normalizasyonu

**Soru:** Piksel değerleri modele hangi ölçekte verilmeli?

| Seçenek | Dönüşüm | Değer aralığı |
|---|---|---|
| Ham piksel | `x` | 0 … 255 |
| Min-max | `x / 255` | 0 … 1 |
| Standardizasyon | `(x / 255 − 0.1306) / 0.3081` | −0.42 … 2.82 (ortalama 0, std 1) |

**Hipotezler:**
- **(a) Ölçek:** Büyük girdi değerleri büyük aktivasyonlara ve gradyanlara yol açar ve eğitimi kararsızlaştırır.
- **(b) Merkezleme:** Ortalaması 0 olan girdilerde, aynı nörona bağlı ağırlıklar aynı adımda farklı yönlere güncellenebilir; bu da yakınsamayı hızlandırır.
- Adam her ağırlığın adım büyüklüğünü kendisi ayarladığı için ölçek sorununu kısmen telafi edebilir.

**Sonuçlar (karar deneyi, Adam):**

| Seçenek | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| Ham piksel (0–255) | %96.90 ± 0.30 | 0.1255 ± 0.0052 | %94.30 ± 0.76 | %98.43 ± 0.31 | 6.3 ± 2.1 | 16.6 ± 0.2 sn |
| Min-max (0–1) | %97.94 ± 0.19 | 0.0789 ± 0.0047 | %95.19 ± 0.44 | %99.51 ± 0.16 | 7.7 ± 1.2 | 16.6 ± 0.0 sn |
| Standardizasyon | %97.75 ± 0.19 | 0.0792 ± 0.0032 | %95.93 ± 0.30 | %99.26 ± 0.22 | 6.0 ± 1.7 | 16.4 ± 0.1 sn |

![Normalizasyon karşılaştırması (Adam)](results/karsilastirma/4a1_normalizasyon/egriler.png)

**Ek analiz (karar için değil):** Bölüm 5.3'te optimizer da değiştirileceği için, normalizasyon seçiminin optimizer'a göre değişip değişmediği düz SGD (lr = 0.1) ile kontrol edildi.

| Seçenek | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| Ham piksel (0–255) | öğrenemedi (hep aynı tahmin, val acc %10.74) | – | – | – | – | – |
| Min-max (0–1) | %97.90 ± 0.13 | 0.0725 ± 0.0017 | %90.74 ± 1.44 | %99.69 ± 0.14 | 12.3 ± 1.5 | 14.4 ± 0.0 sn |
| Standardizasyon | %98.13 ± 0.09 | 0.0719 ± 0.0027 | %93.32 ± 2.21 | %99.93 ± 0.05 | 10.0 ± 1.0 | 14.6 ± 0.1 sn |

![Normalizasyon karşılaştırması (SGD)](results/karsilastirma/4a1b_normalizasyon_sgd/egriler.png)

**Ham piksel + SGD neden hiç öğrenemedi?** Bu modelin içine bakıldığında:
- 1. gizli katmandaki nöronların **%99.6'sı**, 2. gizli katmandakilerin **%100'ü** hiçbir validation görüntüsünde aktif olmuyor (ReLU çıkışı hep 0).
- Model 10.000 validation görüntüsünün hepsine "1" diyor. %10.74, validation setindeki "1" rakamlarının oranı.

0–255 aralığındaki girdilerle ilk adımlardaki gradyanlar çok büyük çıkıyor ve lr = 0.1 ile atılan adımlar neredeyse bütün ReLU nöronlarını kalıcı olarak kapatıyor. Kapalı bir ReLU nöronundan gradyan geçmediği için ağ bir daha toparlanamıyor. Loss NaN olmuyor, rastgele tahmin seviyesinde (ln 10 ≈ 2.30) kalıyor.

**Karar:**
1. **Ham piksel elendi.** Adam ile ~1 puan kötü (anlamlı), SGD ile hiç öğrenemiyor. Hipotez (a) doğrulandı.
2. **Min-max ile standardizasyon doğrulukta ayırt edilemedi.**
   - Adam'da min-max 0.19 puan, SGD'de ise standardizasyon 0.23 puan önde. İkisi de 0.3 puanlık eşiğin altında.
   - Sıralamanın optimizer'a göre değişmesi de farkın gürültüden ibaret olduğunu destekliyor.
3. **İkincil ölçüt (yakınsama hızı) standardizasyonu gösteriyor.** Standardizasyon 1. epoch'ta her iki optimizer'da da önde: Adam'da +0.7 puan, SGD'de +2.6 puan. Bu, hipotez (b) ile uyumlu. Validation loss ve eğitim süresi iki seçenekte eşit.
4. **Seçilen: standardizasyon** (baseline'daki ayar korunuyor).
   - Min-max seçilmedi: doğrulukta ayırt edilemedi ama daha yavaş yakınsıyor.
   - Ham piksel seçilmedi: Adam'da kötü, SGD'de hiç öğrenemiyor.

**Sonraki deneyler için gözlem:** Düz SGD + standardizasyon, Adam + standardizasyondan daha yüksek validation accuracy'ye (%98.13'e karşı %97.75) ve daha düşük validation loss'a (0.072'ye karşı 0.079) ulaştı. SGD'nin validation loss'u Adam'daki gibi yükselmedi. Optimizer kararı Bölüm 5.3'te bu ipucu dikkate alınarak verilecek.

### 5.2 Loss fonksiyonu

**Soru:** Model hangi loss fonksiyonuyla eğitilmeli?

Model her görüntü için 10 skor (logit) `z` üretir. `p = softmax(z)` bu skorların olasılığa çevrilmiş hâlidir, `y` ise doğru sınıfın one-hot gösterimidir.

| Seçenek | Formül | PyTorch |
|---|---|---|
| Cross-entropy | `−log p_doğru` | `nn.CrossEntropyLoss` |
| MSE (softmax + one-hot) | `(1/10) · Σ (p_k − y_k)²` | `softmax` + `F.mse_loss` |
| Hinge (multi-margin) | `(1/10) · Σ max(0, 1 − z_doğru + z_k)`, doğru sınıf hariç bütün `k`'ler için | `nn.MultiMarginLoss` |

- **Cross-entropy:** Doğru sınıfa verilen olasılık düştükçe ceza logaritmik olarak büyür.
- **MSE:** Regresyonda kullanılan kare hatayı olasılıklarla one-hot hedef arasına uygular.
- **Hinge:** SVM'deki loss'un çok sınıflı hâlidir. Olasılık kullanmaz. Doğru sınıfın skoru diğer her sınıfın skorundan en az 1 fazlaysa ceza sıfırdır.

**Hipotezler:**
- **(a) Gradyan büyüklüğü:** İyi bir sınıflandırma loss'u, model yanlışken büyük, doğruyken küçük gradyan üretmelidir.
  - Cross-entropy'de logaritma ile softmax'in türevi birbirini götürür; logit'lere göre gradyan yalnızca `p − y` olur.
  - MSE'de softmax'in türevi gradyanı çarpan olarak küçültür. Model emin bir şekilde yanlış olduğunda bu çarpan sıfıra yaklaşır.
- **(b) Optimizer etkisi:** Adam her ağırlığın adımını o ağırlığın gradyan büyüklüğüne bölerek ölçekler. Bu yüzden küçük gradyanları büyük ölçüde telafi edebilir. Loss'lar arasındaki fark Adam'da küçük, düz SGD'de büyük çıkabilir.

Hipotez (a), tek bir örnek üzerinde kontrol edildi:

| Durum | p(doğru) | Cross-entropy | MSE | Hinge |
|---|---|---|---|---|
| Emin ve yanlış | 0.01 | 4.605 / **1.345** | 0.181 / **0.0176** | 1.351 / **0.949** |
| Kararsız | 0.10 | 2.303 / 0.949 | 0.090 / 0.0190 | 0.900 / 0.949 |
| Emin ve doğru | 0.99 | 0.010 / 0.0105 | 0.000 / 0.0000 | 0.000 / 0.000 |

*Hücreler: loss / gradyan büyüklüğü (loss'un 10 logit'e göre gradyanının normu). 10 sınıflı tek bir örnek kullanıldı; logit'ler doğru sınıfın olasılığı 0.01, 0.10 ve 0.99 olacak şekilde seçildi. Gradyanlar PyTorch autograd ile hesaplandı.*

- **Cross-entropy:** Model ne kadar yanlışsa gradyan o kadar büyük (1.345 > 0.949 > 0.0105).
- **MSE:**
  - Gradyan, cross-entropy'ninkinden 50–76 kat küçük.
  - Emin ve yanlış durumda gradyan, kararsız durumdakinden bile küçük (0.0176 < 0.0190). Model en çok düzeltilmesi gereken yerde en zayıf sinyali alıyor.
- **Hinge:** Gradyan, marj sağlanmadığı sürece sabit (0.949). Modelin ne kadar yanlış olduğunu ayırt etmiyor. Marj sağlandığında gradyan tamamen sıfır oluyor.

**Sonuçlar (karar deneyi, Adam):**

| Seçenek | Val acc | Val loss (her biri kendi loss'u, kıyaslanamaz) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| Cross-entropy | %97.68 ± 0.10 | 0.0782 ± 0.0027 | %95.93 ± 0.30 | %99.21 ± 0.24 | 5.7 ± 2.1 | 17.6 ± 0.2 sn |
| MSE (softmax + one-hot) | %97.89 ± 0.08 | 0.0034 ± 0.0001 | %95.37 ± 0.35 | %99.17 ± 0.22 | 11.3 ± 3.2 | 18.2 ± 0.2 sn |
| Hinge (multi-margin) | %97.45 ± 0.12 | 0.0098 ± 0.0004 | %95.07 ± 0.63 | %98.84 ± 0.32 | 5.3 ± 1.5 | 16.7 ± 0.2 sn |

Her seçeneğin loss'u farklı bir ölçekte olduğu için validation loss değerleri birbiriyle kıyaslanamaz. Bu yüzden grafikte loss yerine train ve validation accuracy çizildi.

![Loss fonksiyonu karşılaştırması (Adam)](results/karsilastirma/4a2_loss/egriler.png)

- **Hinge** en iyi seçenekten 0.45 puan geride. Fark eşiğin üzerinde, yani anlamlı.
- **MSE** sayıca en yüksek sonucu verdi. Ancak cross-entropy ile arasındaki 0.21 puanlık fark eşiğin altında.
- **Cross-entropy** ilk epoch'larda önde: 1. epoch'ta MSE'den 0.56, hinge'den 0.86 puan daha yüksek. Grafikte ~8. epoch'tan sonra eğriler birleşiyor.
- **MSE'nin en iyi epoch'u daha geç** (11.3'e karşı 5.7). MSE ile eğitilen model daha yavaş öğreniyor; train accuracy eğrisi de en altta kalıyor.

**Ek analiz (karar için değil):** Hipotez (b)'yi sınamak ve Bölüm 5.3'te optimizer değişeceği için aynı karşılaştırma düz SGD (lr = 0.1) ile tekrarlandı.

| Seçenek | Val acc | Val loss (her biri kendi loss'u, kıyaslanamaz) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| Cross-entropy | %98.03 ± 0.08 | 0.0713 ± 0.0024 | %93.67 ± 1.54 | %99.90 ± 0.03 | 9.3 ± 0.6 | 15.0 ± 0.1 sn |
| MSE (softmax + one-hot) | %95.62 ± 0.03 | 0.0068 ± 0.0001 | %75.03 ± 2.38 | %96.71 ± 0.05 | 20.0 ± 0.0 | 15.9 ± 0.2 sn |
| Hinge (multi-margin) | %97.86 ± 0.08 | 0.0088 ± 0.0003 | %92.96 ± 0.12 | %99.65 ± 0.01 | 20.0 ± 0.0 | 14.3 ± 0.1 sn |

![Loss fonksiyonu karşılaştırması (SGD)](results/karsilastirma/4a2b_loss_sgd/egriler.png)

- **MSE düz SGD ile çok yavaş öğreniyor.**
  - 1. epoch'ta validation accuracy %75; cross-entropy'de %94.
  - 20 epoch sonunda %95.62 ile cross-entropy'nin 2.41 puan gerisinde kalıyor.
  - Üç seed'in de en iyi epoch'u 20. epoch. Yani model eğitim bittiğinde hâlâ gelişiyordu.

  Hipotez (a) ve (b) birlikte doğrulandı: MSE'nin gradyanı küçük ve Adam bunu telafi ediyordu. Düz SGD'de adım büyüklüğü doğrudan `lr × gradyan` olduğu için küçük gradyan, küçük adım demek.
- **Hinge düz SGD ile cross-entropy'ye yetişiyor** (0.16 puan geride, anlamsız) ve 20. epoch'ta hâlâ gelişiyor. Hinge'in Adam ile neden geride kaldığı bu deneyle belirlenemedi.

**Karar:**
1. **Hinge elendi.** Karar deneyinde (Adam) cross-entropy ve MSE'den anlamlı şekilde kötü.
2. **Cross-entropy ile MSE doğrulukta ayırt edilemedi.** Adam'da MSE 0.21 puan önde; bu, 0.3 puanlık eşiğin altında.
3. **İkincil ölçütler cross-entropy'yi gösteriyor.**
   - Yakınsama hızı: 1. epoch'ta Adam'da 0.56 puan, SGD'de 18.6 puan önde.
   - Sağlamlık: MSE ancak Adam'ın telafisiyle iyi çalışıyor; düz SGD'de 2.41 puan geride kalıyor. Bölüm 5.3'te optimizer değişeceği için optimizer'a bağımlı bir loss seçmek riskli.
   - Eğitim süresi iki seçenekte hemen hemen eşit (17.6 sn'ye karşı 18.2 sn).
4. **Seçilen: cross-entropy** (baseline'daki ayar korunuyor).
   - MSE seçilmedi: Adam'da sayıca en yüksek, ama fark gürültü içinde. Gradyanı zayıf olduğu için Adam olmadan yavaş öğreniyor.
   - Hinge seçilmedi: Adam'da anlamlı şekilde kötü.

**Not – aynı ayarın tekrarları:** Adam + cross-entropy + standardizasyon ayarı Deney 1'de %97.83, Bölüm 5.1'de %97.75, burada %97.68 verdi. SGD + cross-entropy + standardizasyon ayarı ise 5.1'de %98.13, burada %98.03 verdi. Aynı ayarın farklı çalıştırmaları arasında 0.15 puana varan bu farklar, GPU'daki tekrarlanamazlıktan kaynaklanıyor (Bölüm 4.3). Bu da 0.3 puanlık anlamlılık eşiğinin neden gerekli olduğunu gösteriyor.

**Sonraki deneyler için gözlem:** Düz SGD + cross-entropy bu deneyde de Adam + cross-entropy'den yüksek çıktı (%98.03'e karşı %97.68). Böylece SGD iki ayrı deneyde Adam'ın önüne geçmiş oldu. Optimizer kararı Bölüm 5.3'te verilecek.

### 5.3 Optimizer ve learning rate

**Soru:** Ağırlıklar hangi kuralla güncellenmeli ve adımlar ne büyüklükte (learning rate) olmalı?

| Seçenek | Güncelleme kuralı | PyTorch |
|---|---|---|
| SGD | `w ← w − lr · g` | `torch.optim.SGD(lr)` |
| Momentumlu SGD | `v ← 0.9 · v + g`, sonra `w ← w − lr · v` | `torch.optim.SGD(lr, momentum=0.9)` |
| Adam | `w ← w − lr · m / (√s + ε)` | `torch.optim.Adam(lr)`, β₁ = 0.9, β₂ = 0.999 (varsayılan) |

*`g`: mini-batch'in gradyanı. Adam'da `m` gradyanın, `s` gradyanın karesinin hareketli ortalamasıdır (başlangıç düzeltmesiyle).*

- **SGD:** Adım gradyanla orantılı.
- **Momentumlu SGD:** Gradyanlar `v`'de birikir. Gradyan aynı yönü gösterdikçe adım büyür ve düz SGD'deki adımın `1 / (1 − 0.9) = 10` katına çıkar. Yönü adımdan adıma değişen gürültülü bileşenler ise birbirini götürür.
- **Adam:** Her ağırlığın adımı, o ağırlığın gradyan büyüklüğüne bölünür. Bu yüzden adım, gradyan ne kadar büyük ya da küçük olursa olsun ~`lr` civarında kalır.

Aynı gradyan her adımda tekrar verildiğinde ağırlığın toplam değişimi (değerler yukarıdaki kurallardan hesaplandı):

| Gradyan | SGD (lr 0.1)<br>1 adım / 10 adım | Momentumlu SGD (lr 0.01)<br>1 adım / 10 adım | Adam (lr 0.001)<br>1 adım / 10 adım |
|---|---|---|---|
| 0.01 | 0.001 / 0.01 | 0.0001 / 0.0041 | 0.001 / 0.01 |
| 1 | 0.1 / 1 | 0.01 / 0.41 | 0.001 / 0.01 |
| 100 | 10 / 100 | 1 / 41.4 | 0.001 / 0.01 |

Adam'ın adımı gradyandan bağımsız. Bölüm 5.1 ve 5.2'de Adam'ın sorunlu girdi ölçeğini ve MSE'nin küçük gradyanlarını telafi etmesinin sebebi bu.

**Neden learning rate ile birlikte?** Her optimizer'ın uygun learning rate'i farklı. Örneğin Adam'ın varsayılan değeri olan 0.001, düz SGD için çok küçük. Optimizer'ları tek bir ortak learning rate ile karşılaştırmak, en az birini kendine uygun olmayan bir ayarda denemek demek. Bu yüzden her optimizer kendi aralığında, ~3 kat aralıklı değerlerle denendi ve her biri **kendi en iyi learning rate'iyle** karşılaştırıldı.

| Optimizer | Denenen learning rate'ler | Gerekçe |
|---|---|---|
| SGD | 0.01 / 0.03 / 0.1 / 0.3 | Bölüm 5.1–5.2'de lr 0.1 iyi çalıştı; iki yöne de genişletildi |
| Momentumlu SGD | 0.001 / 0.003 / 0.01 / 0.03 | Etkin adım ~10 × lr olduğu için SGD aralığının 1/10'u |
| Adam | 0.00003 / 0.0001 / 0.0003 / 0.001 / 0.003 | Varsayılan değer olan 0.001'in etrafı, aşağı yönde bir değer uzatılmış |

- **Aralık neden aşağı uzatıldı:** En iyi değer denenen aralığın kenarında çıkarsa gerçek en iyi değer aralığın dışında olabilir.
  - Ön denemede Adam'ın en iyi değeri aralığın alt kenarında (0.0001) çıktı. Bu yüzden 0.00003 eklendi.
  - 0.00003'ün belirgin şekilde kötü olduğu görüldü (aşağıdaki tablo).
  - Sonuçta üç optimizer'ın da en iyi değeri aralığın içinde.
- **Toplam:** 13 ayar × 3 seed = 39 eğitim. Diğer bütün ayarlar Bölüm 5.2 sonundaki gibi: standardizasyon, cross-entropy.

**Sonuçlar:**

| Seçenek | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| SGD, lr = 0.01 | %97.32 ± 0.09 | 0.0901 ± 0.0029 | %89.15 ± 0.18 | %98.60 ± 0.02 | 20.0 ± 0.0 | 14.4 ± 0.2 sn |
| SGD, lr = 0.03 | %97.89 ± 0.12 | 0.0720 ± 0.0021 | %91.76 ± 0.57 | %99.84 ± 0.16 | 17.3 ± 3.1 | 14.2 ± 0.1 sn |
| SGD, lr = 0.1 | %98.14 ± 0.09 | 0.0728 ± 0.0017 | %93.94 ± 1.15 | %99.96 ± 0.04 | 11.0 ± 1.7 | 14.3 ± 0.0 sn |
| SGD, lr = 0.3 | %97.66 ± 0.15 | 0.0836 ± 0.0039 | %96.16 ± 0.21 | %99.24 ± 0.28 | 4.7 ± 1.5 | 14.3 ± 0.1 sn |
| Momentumlu SGD, lr = 0.001 | %97.33 ± 0.10 | 0.0903 ± 0.0035 | %89.14 ± 0.06 | %98.58 ± 0.01 | 20.0 ± 0.0 | 14.9 ± 0.2 sn |
| Momentumlu SGD, lr = 0.003 | %97.83 ± 0.02 | 0.0717 ± 0.0025 | %92.10 ± 0.11 | %99.79 ± 0.09 | 15.7 ± 1.5 | 14.7 ± 0.1 sn |
| Momentumlu SGD, lr = 0.01 | %98.10 ± 0.18 | 0.0710 ± 0.0016 | %95.10 ± 0.48 | %99.92 ± 0.10 | 9.7 ± 1.5 | 14.7 ± 0.0 sn |
| Momentumlu SGD, lr = 0.03 | %97.79 ± 0.22 | 0.0835 ± 0.0059 | %96.02 ± 0.70 | %99.41 ± 0.38 | 6.0 ± 1.7 | 14.8 ± 0.0 sn |
| Adam, lr = 0.00003 | %96.54 ± 0.12 | 0.1165 ± 0.0041 | %88.32 ± 0.21 | %97.56 ± 0.11 | 20.0 ± 0.0 | 16.5 ± 0.1 sn |
| Adam, lr = 0.0001 | %97.87 ± 0.09 | 0.0725 ± 0.0017 | %91.78 ± 0.11 | %99.65 ± 0.11 | 17.7 ± 1.2 | 16.5 ± 0.1 sn |
| Adam, lr = 0.0003 | %97.75 ± 0.12 | 0.0735 ± 0.0047 | %94.23 ± 0.18 | %99.51 ± 0.18 | 8.3 ± 1.2 | 16.5 ± 0.1 sn |
| Adam, lr = 0.001 | %97.80 ± 0.06 | 0.0756 ± 0.0060 | %95.96 ± 0.31 | %99.22 ± 0.16 | 5.3 ± 1.5 | 16.5 ± 0.1 sn |
| Adam, lr = 0.003 | %97.25 ± 0.26 | 0.1069 ± 0.0007 | %95.57 ± 0.75 | %98.57 ± 0.34 | 5.3 ± 1.2 | 16.5 ± 0.1 sn |

![Learning rate taraması](results/karsilastirma/4a3_optimizer_lr/lr_tarama.png)

*Her nokta bir ayarın 3 seed ortalaması (en düşük validation loss'lu epoch'ta), hata çubukları ± standart sapma. Yatay eksen logaritmik.*

Her optimizer'ın en iyi learning rate'iyle:

| Seçenek | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|
| SGD, lr = 0.1 | %98.14 ± 0.09 | 0.0728 ± 0.0017 | %93.94 ± 1.15 | %99.96 ± 0.04 | 11.0 ± 1.7 | 14.3 ± 0.0 sn |
| Momentumlu SGD, lr = 0.01 | %98.10 ± 0.18 | 0.0710 ± 0.0016 | %95.10 ± 0.48 | %99.92 ± 0.10 | 9.7 ± 1.5 | 14.7 ± 0.0 sn |
| Adam, lr = 0.0001 | %97.87 ± 0.09 | 0.0725 ± 0.0017 | %91.78 ± 0.11 | %99.65 ± 0.11 | 17.7 ± 1.2 | 16.5 ± 0.1 sn |

![Optimizer karşılaştırması (en iyi learning rate'leriyle)](results/karsilastirma/4a3_optimizer_lr/egriler.png)

- **Momentumlu SGD'nin eğrisi, düz SGD'ninkinin 10 kat sola kaymış hâli.**
  - Düz SGD'de lr ile momentumlu SGD'de lr / 10 her seferinde neredeyse aynı sonucu veriyor: %97.32 ↔ %97.33, %97.89 ↔ %97.83, %98.14 ↔ %98.10, %97.66 ↔ %97.79.
  - Bu, momentum 0.9'un etkin adımı 10 katına çıkardığı öngörüsünü doğruluyor. İki optimizer da en iyi sonucu aynı etkin adımda (0.1) veriyor.
- **Adam learning rate'e en az duyarlı olan.**
  - 0.0001 ile 0.001 arasında (10 kat) sonuç %97.75–97.87 aralığında kalıyor.
  - SGD ve momentumlu SGD'de en iyi değerden 3 kat uzaklaşınca 0.25–0.48 puan kayboluyor.
  - Ancak Adam'ın en iyisi, diğer ikisinin en iyisine ulaşmıyor.
- **Adam'da learning rate, overfitting'in ne zaman başladığını belirliyor.**
  - lr 0.001'de en iyi epoch 5.3; sonrasında validation loss Deney 1'deki gibi yükseliyor.
  - lr 0.0001'de en iyi epoch 17.7. Overfitting gecikiyor ama model 20 epoch sınırına dayanıyor.
- **Düz SGD'de ani düşüş var, momentumlu SGD'de yok.**
  - SGD lr 0.1'de seed 2'nin validation accuracy'si 6. epoch'taki %97.5'ten 8. epoch'ta %96.0'a düşüyor ve 9. epoch'ta %97.9'a dönüyor (grafikte mavi gölgenin genişlediği yer). Aynı düşüş ön denemelerde de görüldü.
  - Momentumlu SGD'de böyle bir düşüş yok.
  - Olası açıklama: İki optimizer'ın etkin adımı aynı. Ancak momentum, gradyanları ~10 adım boyunca biriktirerek tek tek mini-batch'lerin gürültüsünü azaltıyor.

**Karar:**
1. **Birincil ölçüt üç optimizer'ı ayırt etmiyor.**
   - En iyi learning rate'leriyle: SGD %98.14, momentumlu SGD %98.10 (0.04 puan geride), Adam %97.87 (0.27 puan geride).
   - Farkların hepsi 0.3 puanlık eşiğin altında. Adam'ın farkı sınıra yakın.
2. **İkincil ölçütler:**

   | Ölçüt | SGD (lr 0.1) | Momentumlu SGD (lr 0.01) | Adam (lr 0.0001) |
   |---|---|---|---|
   | Yakınsama hızı (1. epoch val acc) | %93.94 ± 1.15 | **%95.10 ± 0.48** | %91.78 ± 0.11 |
   | Val loss (en düşük) | 0.0728 | 0.0710 | 0.0725 |
   | Eğitim süresi (20 epoch) | **14.3 sn** | 14.7 sn | 16.5 sn |
   | Sadelik (ek hiperparametre) | **yok** | momentum (0.9) | β₁, β₂, ε (varsayılan) |
   | Kararlılık | seed 2'de 1.6 puanlık ani düşüş | **pürüzsüz** | **pürüzsüz** |
   | Learning rate'e duyarlılık | 3 kat uzakta 0.25 / 0.48 puan kayıp | 3 kat uzakta 0.27 / 0.31 puan kayıp | **10 kat aralıkta ±0.1 puan** |

   - **Yakınsama hızı** momentumlu SGD'yi gösteriyor: 1. epoch'ta düz SGD'den 1.2, Adam'dan 3.3 puan önde. Aynı sıralama ön denemelerde de görüldü.
   - **Validation loss** farkları standart sapma (~0.002) düzeyinde; ayırt etmiyor.
   - **Eğitim süresi:** Momentumlu SGD düz SGD'den %3 yavaş, Adam ise %15 yavaş. Adam her ağırlık için iki hareketli ortalama tuttuğu için en çok hesabı yapıyor.
   - **Sadelik** düz SGD'nin, **learning rate'e duyarlılık** ise Adam'ın lehine.
3. **Seçilen: momentumlu SGD, lr = 0.01.**
   - Düz SGD seçilmedi: doğrulukta eşit, ama daha yavaş yakınsıyor ve eğitimi daha az kararlı.
   - Adam seçilmedi: doğrulukta eşiğin sınırında geride. En iyi learning rate'inde en yavaş yakınsayan ve en uzun süren seçenek.
   - Ödev şartı açısından: momentumlu SGD, `torch.optim.SGD`'nin momentum parametresiyle kullanılmış hâli. Yani "SGD veya Adam" şartı karşılanıyor.

**Sınırlar:**
- Momentum (0.9) ve Adam'ın β değerleri sabit tutuldu; ayarlanmadı.
- 20 epoch sınırı, düşük learning rate'leri dezavantajlı bırakıyor.
  - SGD lr 0.01 ve momentumlu SGD lr 0.001'de üç seed'in de en iyi epoch'u 20; Adam lr 0.0001'de ortalama 17.7.
  - Bu ayarlar daha uzun eğitimde yükselebilir. Epoch sayısı Bölüm 5.8'de ele alınacak.
- Learning rate eğitim boyunca sabit; zamanla azalan learning rate denenmedi.
- En iyi learning rate batch boyutuna bağlı olabilir (Bölüm 5.6). Optimizer, learning rate ve momentum Bölüm 6'da diğer ayarlarla birlikte yeniden aranacak.

**Not – aynı ayarın tekrarları:** Düz SGD lr 0.1 burada %98.14 verdi (5.1'de %98.13, 5.2'de %98.03). Adam lr 0.001 ise %97.80 verdi (Deney 1'de %97.83, 5.1'de %97.75, 5.2'de %97.68). Tekrarlar arasındaki farklar yine 0.15 puanın içinde.

### 5.4 Derinlik ve genişlik

**Soru:** Kaç gizli katman (derinlik) ve her katmanda kaç nöron (genişlik) olmalı?

**Tasarım:**
- 1–4 gizli katman × 64 / 128 / 256 / 512 / 1024 nöron ızgarası denendi.
- Her gizli katmanda aynı sayıda nöron var. Örneğin "3 katman × 256 nöron" = `784 → 256 → 256 → 256 → 10`. Böylece her model iki sayıyla tanımlanıyor ve derinliğin etkisi genişliğin etkisinden ayrı okunabiliyor.
- Toplam 20 model × 3 seed = 60 eğitim.
- Diğer bütün ayarlar Bölüm 5.3 sonundaki gibi: standardizasyon, cross-entropy, momentumlu SGD (lr = 0.01), batch 64, 20 epoch.
- **1 gizli katmanlı modeller ödev şartını sağlamıyor.** Yalnızca derinliğin ne kazandırdığını görmek için denendi; seçilemezler.

**Parametre sayısı:** `n` girdili, `m` çıktılı bir katmanda `n · m` ağırlık ve `m` bias var. Izgaradaki modeller 50 890 (1 × 64) ile 3 962 890 (4 × 1024) parametre arasında; yani aralarında ~78 kat fark var. Mevcut 256 → 128 modelinde 235 146 parametre var.
- Genişliği 2 katına çıkarmak, ilk katmanı (`784 · w`) 2 kat, iç katmanları (`w · w`) 4 kat büyütüyor.
- Bir katman eklemek `w² + w` parametre ekliyor: 64 nöronda ~4 bin, 1024 nöronda ~1 milyon.

**Hipotezler:**
- **(a) Kapasite:** Çok küçük modeller veriyi yeterince temsil edemeyebilir (underfitting). Büyük modeller ise regularization olmadan train setini daha kolay ezberler.
- **(b) Derinlik:** Tek gizli katmanlı bir ağ, yeterince genişse her sürekli fonksiyona istenen yakınlıkta yaklaşabilir (evrensel yaklaşım teoremi; Cybenko, 1989; Hornik, 1991). Teorem bunun için kaç nöron gerektiğini söylemiyor. Derinlik, özellikleri katman katman birleştirerek aynı işi daha az parametreyle yapabilir. MNIST görece kolay bir problem olduğu için derinliğin etkisinin küçük olması beklendi.
- **(c) Süre:** Küçük modellerde GPU kapasitesinin çok altında çalışıyor. Bu yüzden süre nöron sayısından çok, arka arkaya yapılan işlem sayısına, yani katman sayısına bağlı olabilir.

**Seçim kuralı (sonuçlardan önce belirlendi):**
1. Birincil ölçüt öncekilerle aynı. Seçilebilir modeller en az 2 gizli katmanlı olanlar (ödev şartı).
2. Seçilebilir modellerin en iyisine 0.3 puandan yakın olanlar "eşit derecede iyi" sayılır.
3. Bunlardan **en az parametreli** olan seçilir.
   - Fark gürültü düzeyindeyse, büyük modelin daha iyi olduğuna dair bir kanıt yok. Küçük model daha hızlı eğitiliyor ve ezberleme kapasitesi daha düşük.
   - Bu, model seçimindeki **"bir standart hata kuralı"**nın benzeri: hatası en iyi modelinkinden bir standart hata içinde kalan modellerin en sadesi seçilir (Hastie, Tibshirani ve Friedman, *The Elements of Statistical Learning*, 2009, Bölüm 7.10). Bizim 0.3 puanlık eşiğimiz aynı rolü oynuyor.
   - Bölüm 5.0'daki ikincil ölçütlerden "sadelik"in bu deneydeki karşılığı parametre sayısı. Seçenekler tam da boyutta ayrıldığı için belirleyici ölçüt o. 1. epoch accuracy bu deneyde adil bir hız ölçüsü değil: büyük model bir epoch'ta daha çok öğrenebilir, ama o epoch'u daha pahalıya yapar.

**Sonuçlar:**

| Seçenek | Parametre | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|---|
| 1 katman × 64 nöron | 50 890 | %97.36 ± 0.06 | 0.0912 ± 0.0027 | %94.48 ± 0.21 | %99.26 ± 0.21 | 8.3 ± 1.2 | 11.3 ± 0.1 sn |
| 1 katman × 128 nöron | 101 770 | %97.91 ± 0.11 | 0.0756 ± 0.0015 | %94.93 ± 0.27 | %99.84 ± 0.10 | 11.0 ± 1.7 | 11.2 ± 0.0 sn |
| 1 katman × 256 nöron | 203 530 | %97.94 ± 0.11 | 0.0684 ± 0.0023 | %95.22 ± 0.25 | %99.82 ± 0.14 | 9.3 ± 2.1 | 11.4 ± 0.1 sn |
| 1 katman × 512 nöron | 407 050 | %98.12 ± 0.03 | 0.0650 ± 0.0015 | %95.41 ± 0.23 | %99.95 ± 0.02 | 10.7 ± 0.6 | 11.3 ± 0.0 sn |
| 1 katman × 1024 nöron | 814 090 | %98.20 ± 0.05 | 0.0628 ± 0.0009 | %95.69 ± 0.29 | %99.95 ± 0.04 | 9.7 ± 1.5 | 11.3 ± 0.1 sn |
| 2 katman × 64 nöron | 55 050 | %97.41 ± 0.15 | 0.0924 ± 0.0029 | %94.46 ± 0.38 | %99.44 ± 0.09 | 9.3 ± 0.6 | 13.5 ± 0.1 sn |
| 2 katman × 128 nöron | 118 282 | %97.60 ± 0.10 | 0.0780 ± 0.0029 | %94.91 ± 0.34 | %99.32 ± 0.15 | 6.3 ± 0.6 | 14.8 ± 0.1 sn |
| **2 katman × 256 nöron** | 269 322 | %97.99 ± 0.11 | 0.0700 ± 0.0016 | %95.21 ± 0.33 | %99.70 ± 0.31 | 7.7 ± 2.5 | 14.8 ± 0.1 sn |
| 2 katman × 512 nöron | 669 706 | %98.23 ± 0.20 | 0.0672 ± 0.0002 | %95.44 ± 0.30 | %99.92 ± 0.11 | 9.3 ± 2.1 | 14.7 ± 0.0 sn |
| 2 katman × 1024 nöron | 1 863 690 | %98.21 ± 0.17 | 0.0653 ± 0.0021 | %95.73 ± 0.32 | %99.92 ± 0.09 | 8.0 ± 2.0 | 14.8 ± 0.0 sn |
| 3 katman × 64 nöron | 59 210 | %97.28 ± 0.18 | 0.0913 ± 0.0054 | %93.60 ± 1.25 | %98.74 ± 0.30 | 6.0 ± 1.0 | 15.4 ± 0.1 sn |
| 3 katman × 128 nöron | 134 794 | %97.70 ± 0.10 | 0.0804 ± 0.0012 | %94.61 ± 0.69 | %99.42 ± 0.15 | 7.3 ± 1.2 | 18.1 ± 0.0 sn |
| 3 katman × 256 nöron | 335 114 | %97.81 ± 0.22 | 0.0743 ± 0.0023 | %94.97 ± 0.48 | %99.40 ± 0.28 | 6.0 ± 1.0 | 18.1 ± 0.1 sn |
| 3 katman × 512 nöron | 932 362 | %97.92 ± 0.11 | 0.0708 ± 0.0016 | %95.37 ± 0.36 | %99.53 ± 0.23 | 5.3 ± 1.5 | 18.3 ± 0.1 sn |
| 3 katman × 1024 nöron | 2 913 290 | %98.05 ± 0.11 | 0.0703 ± 0.0030 | %95.53 ± 0.37 | %99.74 ± 0.11 | 6.0 ± 1.0 | 18.1 ± 0.0 sn |
| 4 katman × 64 nöron | 63 370 | %97.22 ± 0.22 | 0.0954 ± 0.0062 | %92.84 ± 1.97 | %98.93 ± 0.19 | 7.3 ± 1.5 | 17.6 ± 0.1 sn |
| 4 katman × 128 nöron | 151 306 | %97.45 ± 0.15 | 0.0863 ± 0.0051 | %93.73 ± 1.76 | %99.22 ± 0.04 | 6.3 ± 0.6 | 21.8 ± 0.0 sn |
| 4 katman × 256 nöron | 400 906 | %97.75 ± 0.09 | 0.0784 ± 0.0011 | %94.50 ± 1.20 | %99.37 ± 0.25 | 5.7 ± 1.5 | 21.9 ± 0.1 sn |
| 4 katman × 512 nöron | 1 195 018 | %97.82 ± 0.10 | 0.0762 ± 0.0031 | %95.19 ± 0.70 | %99.48 ± 0.31 | 6.0 ± 1.7 | 21.8 ± 0.0 sn |
| 4 katman × 1024 nöron | 3 962 890 | %97.82 ± 0.19 | 0.0762 ± 0.0026 | %95.22 ± 0.94 | %99.54 ± 0.29 | 5.7 ± 2.1 | 22.0 ± 0.1 sn |

![Derinlik ve genişlik ızgarası](results/karsilastirma/4a4_derinlik_genislik/izgara.png)

*Her nokta bir modelin 3 seed ortalaması (en düşük validation loss'lu epoch'ta), hata çubukları ± standart sapma. Yatay eksen logaritmik; her adım 2 kat.*

Her derinliğin en iyi modeli:

| Seçenek | Parametre | Val acc | Val loss (en düşük) | 1. epoch val acc | Train acc | En iyi epoch | Eğitim süresi (20 epoch) |
|---|---|---|---|---|---|---|---|
| 1 katman × 1024 nöron | 814 090 | %98.20 ± 0.05 | 0.0628 ± 0.0009 | %95.69 ± 0.29 | %99.95 ± 0.04 | 9.7 ± 1.5 | 11.3 ± 0.1 sn |
| 2 katman × 512 nöron | 669 706 | %98.23 ± 0.20 | 0.0672 ± 0.0002 | %95.44 ± 0.30 | %99.92 ± 0.11 | 9.3 ± 2.1 | 14.7 ± 0.0 sn |
| 3 katman × 1024 nöron | 2 913 290 | %98.05 ± 0.11 | 0.0703 ± 0.0030 | %95.53 ± 0.37 | %99.74 ± 0.11 | 6.0 ± 1.0 | 18.1 ± 0.0 sn |
| 4 katman × 512 nöron | 1 195 018 | %97.82 ± 0.10 | 0.0762 ± 0.0031 | %95.19 ± 0.70 | %99.48 ± 0.31 | 6.0 ± 1.7 | 21.8 ± 0.0 sn |

![Her derinliğin en iyi modeli](results/karsilastirma/4a4_derinlik_genislik/egriler.png)

- **Genişlik önemli, ama bir yere kadar.** 2 katmanda doğruluk 64 nörondan 512 nörona çıkarken %97.41'den %98.23'e yükseliyor; 1024'te artık artmıyor (%98.21). Diğer derinliklerde de aynı eğilim var. Hipotez (a) doğrulandı: küçük modeller anlamlı şekilde kötü.
- **Derinlik bu problemde bir şey kazandırmadı.**
  - Aynı genişlikte 1 ve 2 katman en iyi. 3 ve 4 katman çoğu genişlikte daha kötü: 4 × 256 %97.75, 2 × 256 %97.99.
  - Tek gizli katmanlı 1 × 1024 (%98.20), ızgaranın en iyileri arasında. Hipotez (b) ile uyumlu: MNIST için tek bir geniş katman yeterli.
  - Derin modellerin validation loss'u da daha yüksek: 4 × 256'da 0.078, 2 × 256'da 0.070.
- **Derin modeller yavaş ve seed'e bağlı başlıyor.**
  - 1. epoch'ta 4 × 64 %92.84 ± 1.97, 1 × 64 %94.48 ± 0.21.
  - Sebep, PyTorch'un varsayılan başlangıç ağırlıkları. `nn.Linear` ağırlıkları `[−1/√n, 1/√n]` aralığından seçiliyor (varyans `1/(3n)`), ReLU da sinyalin yaklaşık yarısını sıfırlıyor. Hesaba göre sinyalin büyüklüğü her katmanda ~√6 ≈ 2.4 kat küçülüyor.
  - Eğitilmemiş bir 4 × 256 modelde 1 000 validation görüntüsüyle ölçüldü: katman çıkışlarının standart sapması 0.56 → 0.23 → 0.10 → 0.05 şeklinde küçülüyor. ReLU için önerilen He başlangıcıyla (He vd., 2015) ise her katmanda ~1.4'te sabit kalıyor.
  - Zayıf sinyal, küçük gradyan ve yavaş başlangıç demek. Bu ödevde başlangıç yöntemi ayrıca karşılaştırılmadı.
- **Süre katman sayısıyla artıyor, genişlikle neredeyse hiç değişmiyor.** 1 katmanlı modellerin hepsi ~11.3 sn, 2 katmanlılar ~14.8 sn, 3 katmanlılar ~18 sn, 4 katmanlılar ~22 sn sürüyor; 64 nöron da 1024 nöron da aynı süreyi alıyor. Hipotez (c) doğrulandı.
- **4 katmanlı modelin validation loss eğrisi** diğerlerinden yüksekte ve daha dalgalı seyrediyor.

**Karar:**
1. **En iyi seçilebilir model:** 2 × 512 (%98.23).
2. **Eşik içindeki modeller:** 2 × 256 (%97.99), 2 × 512, 2 × 1024 (%98.21) ve 3 × 1024 (%98.05). 3 × 512 (%97.92) 0.31 puanla eşiğin hemen dışında kaldı.
3. **Seçilen: 2 gizli katman × 256 nöron** (`784 → 256 → 256 → 10`, 269 322 parametre). Parametre sayısı 2 × 512'nin %40'ı, 2 × 1024'ün yedide biri.
   - **Daha büyük modeller seçilmedi:** Doğrulukta ayırt edilemiyorlar, ama 2.5–11 kat daha fazla parametreleri var.
   - **64 ve 128 nöronlu modeller seçilmedi:** Anlamlı şekilde kötüler (2 × 128, 0.63 puan geride).
   - **3 ve 4 katman seçilmedi:** Aynı genişlikte kazanç getirmiyorlar; daha yavaşlar ve val loss'ları daha yüksek.
   - **1 katman seçilemez:** ödev şartı.
   - Mevcut 256 → 128 modelinden tek farkı, ikinci katmanın 128 yerine 256 nöron olması.

**Sınırlar:**
- 2 × 256'nın farkı 0.24 puan, yani eşiğe yakın; tekrarlanan bir çalıştırmada 2 × 512 seçilebilirdi. Kural önceden belirlendiği için sonucu olduğu gibi aldık.
- Bütün modeller 256 → 128 modelinde seçilen learning rate (0.01) ile eğitildi. Çok farklı boyuttaki modellerin en iyi learning rate'i farklı olabilir.
- Başlangıç ağırlıkları PyTorch'un varsayılanı; He başlangıcı derin modellere yardım edebilirdi.
- Regularization yok. Büyük modeller regularization ile (Bölüm 5.7) daha iyi sonuç verebilir. Mimari, Bölüm 6'da diğer ayarlarla birlikte yeniden aranacak.

## 6. Ortak Hiperparametre Araması
_(Aşama 4B–4C'de doldurulacak)_

## 7. Final Model ve Test Sonucu
_(Aşama 5'te doldurulacak)_

## 8. Kendi El Yazım ile Değerlendirme
_(Aşama 6'da doldurulacak)_

## 9. Genel Değerlendirme
_(Aşama 7'de doldurulacak)_
