# Full Dataset Training Analizi (lanelm_v4_full)

## Test Sonuçları (CULane Full Test Set - 34680 images)

### Metrikler (IoU Thresholds)

#### IoU 0.1 (Düşük eşik - daha toleranslı)
- **F1: 0.5214** (kabul edilebilir ama ideal değil)
- **Precision: 0.4578** (düşük - çok fazla false positive)
- **Recall: 0.6054** (orta - bazı şeritler kaçırılıyor)
- **TP: 63503, FP: 75217, FN: 41383**

#### IoU 0.5 (Standart eşik - CULane standardı)
- **F1: 0.0124** ⚠️ **ÇOK KÖTÜ!**
- **Precision: 0.0109** (neredeyse 0 - çok fazla false positive)
- **Recall: 0.0144** (neredeyse 0 - neredeyse hiçbir şerit doğru tespit edilmiyor)
- **TP: 1515, FP: 137205, FN: 103371**

#### IoU 0.75 (Yüksek eşik - çok katı)
- **F1: 0.0005** ⚠️ **NEREDEYSE 0!**
- **Precision: 0.0004** (neredeyse 0)
- **Recall: 0.0006** (neredeyse 0)
- **TP: 59, FP: 138661, FN: 104827**

## Kritik Sorunlar

### 1. **Koordinat Hizalaması Sorunu**
- IoU 0.1'de F1=0.52 (kabul edilebilir) ama IoU 0.5'te F1=0.01 (felaket)
- Bu, **tahminlerin GT'ye yakın ama tam hizalı olmadığını** gösteriyor
- Muhtemelen:
  - Normalizasyon hatası (resized → original space conversion)
  - Smoothing sonrası koordinat kayması
  - Decode mantığında hata

### 2. **Çok Fazla False Positive (FP)**
- IoU 0.5: **FP=137205** (çok fazla yanlış tahmin)
- Model "hallucination" yapıyor - olmayan şeritleri tespit ediyor
- Hallucination Removal (HR) yeterince güçlü değil veya uygulanmıyor

### 3. **Çok Fazla False Negative (FN)**
- IoU 0.5: **FN=103371** (çok fazla kaçırılan şerit)
- Model mevcut şeritleri tespit edemiyor
- Muhtemelen:
  - Model yeterince genelleştiremiyor
  - Overfitting (training loss düşük ama test kötü)
  - Visual encoder yeterince güçlü değil

### 4. **Kategori Bazlı Performans**
- **test0_normal**: F1@0.5=0.0264 (en iyi ama yine de çok kötü)
- **test1_crowd**: F1@0.5=0.0069 (kalabalık sahnelerde çok kötü)
- **test2_hlight**: F1@0.5=0.0088 (yüksek ışıkta kötü)
- **test3_shadow**: F1@0.5=0.0085 (gölgede kötü)
- **test4_noline**: F1@0.5=0.0073 (çizgi yokken kötü)
- **test5_arrow**: F1@0.5=0.0160 (ok işaretlerinde nispeten daha iyi)
- **test6_curve**: F1@0.5=0.0140 (eğri şeritlerde kötü)
- **test7_cross**: F1@0.5=0.0000 (kavşaklarda hiç çalışmıyor!)
- **test8_night**: F1@0.5=0.0056 (gece sahnelerinde çok kötü)

## Olası Nedenler

### 1. **Training-Validation Mismatch**
- Training'de görselleştirmeler iyi görünüyor ama test'te kötü
- Bu, **training ve test arasında decode/normalization farkı** olduğunu gösteriyor
- `train_lanelm_v4_fixed.py`'deki `visualize` fonksiyonu ile test'teki `LaneLMDetector.predict` farklı olabilir

### 2. **Overfitting**
- Model training set'ini ezberlemiş ama genelleştiremiyor
- Full dataset training yeterince uzun sürmemiş olabilir
- Learning rate çok yüksek veya scheduler yanlış ayarlanmış olabilir

### 3. **Posterior Collapse (Kısmen)**
- Model hala visual bilgiyi tam kullanmıyor olabilir
- Cross-attention uniform olabilir (debug edilmeli)
- Visual encoder yeterince güçlü değil

### 4. **Normalization/Coordinate Conversion Hatası**
- Resized space (800x320) → Original space (1640x590) conversion'da hata olabilir
- `coords_to_lane_normalized` fonksiyonunda sorun olabilir
- Smoothing sonrası koordinatlar kaymış olabilir

## Öneriler

### 1. **100-Image Subset ile Hızlı İterasyon**
- Full dataset training çok zaman alıyor
- 100-image train + 100-image test subset'leri oluştur
- Hızlı iterasyon yap, sorunları çöz, sonra full dataset'e ölçekle

### 2. **Training-Test Decode Alignment**
- `train_lanelm_v4_fixed.py`'deki `visual_first_decode` ile `LaneLMDetector.predict`'teki `autoregressive_decode` **tamamen aynı** olmalı
- Normalization mantığı birebir aynı olmalı
- Smoothing parametreleri aynı olmalı

### 3. **Hallucination Removal Güçlendirme**
- HR algoritmasını gözden geçir
- Daha agresif filtreleme yap
- Geometric constraints ekle

### 4. **Visual Encoder Güçlendirme**
- Full FPN yerine P5-only kullanılıyor (V5'te adaptive pooling var)
- Visual token sayısını artır (adaptive pooling'den önce)
- 2D PE'yi güçlendir

### 5. **Training Strategy İyileştirme**
- Learning rate'i düşür
- Epoch sayısını artır
- Scheduled sampling'i daha agresif yap
- AR rollout loss weight'ini artır

## Sonraki Adımlar

1. ✅ **100-image subset'leri oluştur** (train_100.txt, test_100.txt)
2. ✅ **Training script'ini 100-image ile çalışacak şekilde güncelle**
3. ⏳ **Test script'ini 100-image ile çalışacak şekilde güncelle**
4. ⏳ **Training-test decode alignment'ı doğrula**
5. ⏳ **100-image ile hızlı iterasyon yap**
6. ⏳ **Sorunları çöz**
7. ⏳ **Full dataset'e ölçekle**








