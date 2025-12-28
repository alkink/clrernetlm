# Debug Analizi - Detaylı Yorum

## Kritik Bulgular

### 1. ✅ Training vs Test Decode: BİREBİR AYNI

**Resized Space'de (800x320):**
- Tüm lane'ler için X ve Y koordinatları **tamamen aynı** (0.0 px fark)
- Lane 0: 23 points, X[48.8, 348.4], Y[24.0, 208.0] - **BİREBİR AYNI**
- Lane 1: 37 points, X[261.0, 377.9], Y[24.0, 312.0] - **BİREBİR AYNI**
- Lane 2: 37 points, X[400.2, 680.1], Y[24.0, 312.0] - **BİREBİR AYNI**
- Lane 3: 17 points, X[434.1, 780.0], Y[24.0, 152.0] - **BİREBİR AYNI**

**Sonuç:** 
- ✅ Decode mantığı tamamen aynı
- ✅ Tokenization aynı
- ✅ Smoothing aynı
- ✅ Resized space'de hiçbir fark yok

### 2. ✅ Normalization: DOĞRU ÇALIŞIYOR

**Test Point:**
- Resized: X=400.0, Y=160.0
- Original: X=820.0, Y=430.0
- Normalized: X=0.5000, Y=0.7288
- Scale factors: X=2.05, Y=1.0

**Normalized Coordinates:**
- Lane 0: X[0.0610, 0.4355], Y[0.4983, 0.8102]
- Lane 1: X[0.3263, 0.4724], Y[0.4983, 0.9864]
- Lane 2: X[0.5002, 0.8501], Y[0.4983, 0.9864]
- Lane 3: X[0.5427, 0.9751], Y[0.4983, 0.7153]

**Sonuç:**
- ✅ Normalization matematiksel olarak doğru
- ✅ Tüm değerler [0, 1) aralığında
- ✅ Scale faktörleri doğru (X=2.05, Y=1.0)

### 3. ⚠️ SORUN: Normalization Sonrası

**Gözlem:**
- Resized space'de training ve test **birebir aynı**
- Normalization **doğru çalışıyor**
- Ama test sonuçları **çok kötü** (F1=0.0132)

**Bu demek ki sorun:**
1. **GT Loading Farkı:** Training'de GT resized space'de, test'te GT original space'den yükleniyor
2. **CULaneMetric'in GT ve Prediction Yüklemesi:** Farklı format/space'de olabilir
3. **Spline Interpolation:** Normalized space'de spline yanlış çalışıyor olabilir
4. **IoU Hesaplama:** CULaneMetric'in IoU hesaplaması yanlış olabilir

## Detaylı Analiz

### Normalized Y Değerleri Analizi

**Gözlem:** Tüm lane'lerin Y değerleri **0.4983'ten başlıyor**

**Hesaplama:**
- Resized Y=24.0 (en üst)
- Original Y = 24.0 * 1.0 + 270 = 294.0
- Normalized Y = 294.0 / 590.0 = **0.4983** ✓

**Bu doğru!** Crop offset (y_min=270) doğru uygulanıyor.

### Normalized X Değerleri Analizi

**Lane 0:**
- Resized X: [48.8, 348.4]
- Normalized X: [0.0610, 0.4355]
- Kontrol: 48.8 * 2.05 / 1640 = 0.0610 ✓
- Kontrol: 348.4 * 2.05 / 1640 = 0.4355 ✓

**Bu da doğru!**

## Olası Sorunlar

### 1. GT Loading Farkı (EN OLASI)

**Training:**
- GT pipeline'dan geliyor (resized space'de, 800x320)
- Görselleştirmede resized space'de çiziliyor

**Test:**
- GT `.lines.txt` dosyasından yükleniyor (original space'de, 1640x590)
- CULaneMetric GT'yi normalize ediyor
- Ama normalization farklı olabilir!

**Kontrol Edilmeli:**
- CULaneMetric GT'yi nasıl yüklüyor?
- GT normalization'ı prediction normalization'ı ile aynı mı?

### 2. Spline Interpolation Sorunu

**Normalized space'de:**
- `Lane` class'ı spline interpolation yapıyor
- Y değerleri strictly increasing olmalı
- Ama bazı durumlarda spline başarısız olabilir

**Kontrol Edilmeli:**
- Normalized space'de spline başarılı mı?
- Interpolated points doğru mu?

### 3. IoU Hesaplama Sorunu

**CULaneMetric:**
- GT ve prediction'ı normalized space'de karşılaştırıyor
- Ama format farklı olabilir
- IoU hesaplaması yanlış olabilir

**Kontrol Edilmeli:**
- CULaneMetric'in IoU hesaplaması doğru mu?
- GT ve prediction format'ı aynı mı?

## Sonraki Adımlar

### 1. GT Loading Kontrolü
- CULaneMetric'in GT yükleme mantığını incele
- GT normalization'ını kontrol et
- Prediction normalization'ı ile karşılaştır

### 2. Spline Interpolation Kontrolü
- Normalized space'de spline başarılı mı?
- Interpolated points doğru mu?
- Spline hataları var mı?

### 3. CULaneMetric Debug
- CULaneMetric'in IoU hesaplamasını debug et
- GT ve prediction'ı logla
- IoU hesaplama adımlarını incele

### 4. Görselleştirme Ekle
- Normalized space'de GT ve prediction'ı görselleştir
- Overlay yap ve hizalı mı kontrol et

## Öneriler

### 1. CULaneMetric GT Loading Debug
```python
# CULaneMetric'in GT yükleme mantığını incele
# GT normalization'ını kontrol et
# Prediction normalization'ı ile karşılaştır
```

### 2. Normalized Space Görselleştirme
```python
# Normalized space'de GT ve prediction'ı görselleştir
# Overlay yap ve hizalı mı kontrol et
# Fark varsa, normalization'da sorun var demektir
```

### 3. Spline Interpolation Test
```python
# Normalized space'de spline başarılı mı?
# Interpolated points doğru mu?
# Spline hataları var mı?
```

## Sonuç

**Kesin Bulgular:**
- ✅ Resized space'de training ve test **birebir aynı**
- ✅ Normalization **matematiksel olarak doğru**
- ⚠️ Sorun **normalization sonrası** bir yerde

**En Olası Sorun:**
- **GT Loading Farkı:** Training'de resized GT, test'te original GT
- **CULaneMetric GT Normalization:** Farklı normalization olabilir
- **Spline Interpolation:** Normalized space'de spline hatası

**Sonraki Adım:**
- CULaneMetric'in GT yükleme ve normalization mantığını incele
- Normalized space'de GT ve prediction'ı görselleştir
- Spline interpolation'ı test et








