# Root Cause Analysis - Training vs Test Mismatch

## Debug Sonuçları Özeti

### ✅ Kesin Bulgular
1. **Resized Space'de Training ve Test BİREBİR AYNI**
   - Tüm lane'ler için X ve Y koordinatları tamamen aynı (0.0 px fark)
   - Decode mantığı, tokenization, smoothing - hepsi aynı

2. **Normalization Matematiksel Olarak Doğru**
   - Scale faktörleri doğru (X=2.05, Y=1.0)
   - Normalized coordinates [0, 1) aralığında
   - Test point doğru normalize ediliyor

### ⚠️ Sorun: Normalization Sonrası

**Gözlem:**
- Resized space'de birebir aynı
- Normalization doğru
- Ama test sonuçları çok kötü (F1=0.0132)

## CULaneMetric Mantığı Analizi

### 1. Prediction String Oluşturma (`get_prediction_string`)

```python
# Normalized Y değerleri (0-1 aralığında)
ys = np.arange(0, self.ori_h, self.y_step) / self.ori_h  # [0, 0.0034, 0.0068, ...]

# Lane class'ı callable - normalized Y'den normalized X hesaplıyor
xs = lane(ys_in_range)  # Normalized X (0-1)

# Original space'e çevir
xs = xs * self.ori_w  # Original X (0-1640)
lane_ys = ys_in_range[valid_mask] * self.ori_h  # Original Y (0-590)

# Prediction string'e yaz (original space'de)
lane_str = " ".join(["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)])
```

**Önemli:** Prediction string **original space'de** (1640x590) koordinatlar içeriyor.

### 2. GT Yükleme (`load_culane_img_data`)

```python
# GT .lines.txt dosyasından yükleniyor
# Format: "x0 y0 x1 y1 x2 y2 ..."
# Koordinatlar: Original space'de (1640x590)
```

**Önemli:** GT de **original space'de** (1640x590) koordinatlar içeriyor.

### 3. IoU Hesaplama (`culane_metric`)

```python
# Prediction ve GT'yi interpolate et (original space'de)
interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred], dtype=object)
interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno], dtype=object)

# Original space'de IoU hesapla
ious = discrete_cross_iou(interp_pred, interp_anno, width=30, img_shape=(590, 1640, 3))
```

**Önemli:** IoU hesaplama **original space'de** (590x1640) yapılıyor.

## Olası Sorunlar

### 1. **Lane Class Spline Interpolation Sorunu** (EN OLASI)

**Sorun:**
- `Lane` class'ı normalized space'de (0-1) points alıyor
- `__call__` fonksiyonu normalized Y'den normalized X hesaplıyor
- Ama spline interpolation başarısız olabilir

**Kontrol:**
- `coords_to_lane_normalized` fonksiyonu normalized points'i `Lane` class'ına veriyor
- `Lane.__init__` spline oluşturuyor: `InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))`
- Eğer points'ler düzgün sıralanmamışsa veya duplicate Y değerleri varsa spline başarısız olabilir

**Test:**
- Normalized space'de spline başarılı mı?
- `Lane(ys_in_range)` doğru X değerleri döndürüyor mu?

### 2. **Y Range Filtreleme Sorunu**

**Sorun:**
- `get_prediction_string`'de `ys_in_range = ys[(ys >= lane_min_y) & (ys <= lane_max_y)]` ile Y range filtreleniyor
- Ama `lane_min_y` ve `lane_max_y` normalized space'de (0-1)
- Eğer bu değerler yanlışsa, yanlış Y değerleri için interpolasyon yapılabilir

**Kontrol:**
- `lane_min_y` ve `lane_max_y` doğru mu?
- Y range filtresi doğru çalışıyor mu?

### 3. **Extrapolation Sorunu**

**Sorun:**
- `Lane` class'ı `min_y` ve `max_y` dışındaki Y değerleri için `invalid_value=-2.0` döndürüyor
- Ama `get_prediction_string`'de bu değerler filtreleniyor mu?

**Kontrol:**
- `invalid_value` doğru filtreleniyor mu?
- Extrapolation yapılıyor mu?

### 4. **Coordinate Conversion Hatası**

**Sorun:**
- `coords_to_lane_normalized` fonksiyonu resized space'den normalized space'e çeviriyor
- Ama bu conversion'da bir hata olabilir (ama debug'da doğru görünüyor)

**Kontrol:**
- Normalized coordinates doğru mu?
- `Lane` class'ına verilen points doğru mu?

## Debug Stratejisi

### 1. **Lane Class Spline Test**
```python
# Normalized space'de bir Lane oluştur
points_norm = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]])
lane = Lane(points_norm)

# Test Y değerleri (normalized)
test_ys = np.array([0.5, 0.6, 0.7, 0.8])

# X değerlerini hesapla
xs = lane(test_ys)

# Beklenen değerlerle karşılaştır
print(f"Expected: [0.1, 0.2, 0.3, 0.4]")
print(f"Got: {xs}")
```

### 2. **get_prediction_string Debug**
```python
# Prediction string oluştur
pred_string = get_prediction_string(lanes)

# İlk birkaç satırı kontrol et
lines = pred_string.split('\n')
for i, line in enumerate(lines[:3]):
    coords = line.split()
    xs = [float(coords[j]) for j in range(0, len(coords), 2)]
    ys = [float(coords[j+1]) for j in range(0, len(coords), 2)]
    print(f"Line {i}: X range: [{min(xs):.1f}, {max(xs):.1f}], Y range: [{min(ys):.1f}, {max(ys):.1f}]")
    
# Original space'de (1640x590) olmalı
# X: [0, 1640), Y: [0, 590)
```

### 3. **GT vs Prediction Overlay (Original Space)**
```python
# GT ve prediction'ı original space'de görselleştir
# Overlay yap ve hizalı mı kontrol et
```

## Sonraki Adımlar

1. ✅ **Lane Class Spline Test** - Normalized space'de spline doğru çalışıyor mu?
2. ⏳ **get_prediction_string Debug** - Prediction string doğru format'ta mı?
3. ⏳ **GT vs Prediction Overlay** - Original space'de hizalı mı?
4. ⏳ **IoU Hesaplama Debug** - IoU hesaplama doğru mu?

## En Olası Sorun

**Lane Class Spline Interpolation:**
- Normalized space'de spline başarısız olabilir
- Points'ler düzgün sıralanmamış olabilir
- Duplicate Y değerleri olabilir
- Spline extrapolation yapıyor olabilir (min_y/max_y dışında)

**Çözüm:**
- `coords_to_lane_normalized` fonksiyonunda points'leri düzgün sırala
- Duplicate Y değerlerini kaldır
- Spline başarısız olursa hata ver
- `get_prediction_string`'de extrapolation yapma








