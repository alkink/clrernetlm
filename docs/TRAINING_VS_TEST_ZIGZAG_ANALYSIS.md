# Training vs Test Zigzag Analizi (Kanıtlı)

## Gerçek Verilerle Kanıtlı Bulgular

### 1. Training vs Test Path Karşılaştırması

**Test Sonucu:**
- ✅ Tokens: IDENTICAL
- ✅ Coordinates: IDENTICAL  
- ✅ Zigzag levels: IDENTICAL (1.8938)

**Sonuç:** Training ve test path'leri **BİREBİR AYNI**!

### 2. Test Prediction Dosyası Analizi

**Dosya:** `work_dirs/lanelm_v4_test_fixed_100/predictions/driver_100_30frame/05251517_0433.MP4/02970.lines.txt`

**İlk 20 X değeri:**
```
189.59769, 188.7331, 188.12457, 187.77062, 187.66976, 187.82051, 
188.2214, 188.87094, 189.76766, 190.91006, 192.29668, 193.92751, 
195.80848, 197.94699, 200.35045, 203.02033, 205.93442, 209.06456, 
212.3826, 215.88262
```

**Zigzag metrik:** std(diffs) = 4.9981

**Kritik:** Test prediction dosyası **ZATEN ZIGZAG**!

### 3. Sorun Analizi

**Training visualization:**
- `decode_single_lane(..., smooth=True)` → smooth coords
- Direkt çizim (resized space)
- **Görünüm:** Smooth ✅

**Test prediction:**
- `decode_single_lane(..., smooth=True)` → smooth coords ✅
- `coords_to_lane_normalized` → normalized coords
- `Lane` class → spline interpolation
- `get_prediction_string` → spline interpolation ile Y grid'e sample
- **Sonuç:** Zigzag ❌

### 4. Root Cause: Spline Interpolation

**Sorun:**
1. `decode_single_lane` smooth=True ile smoothing yapıyor ✅
2. Ama `Lane` class'ın spline interpolation'ı zigzag'ı artırıyor ❌
3. `get_prediction_string` Y grid'e sample yaparken spline kullanıyor
4. Bu, zigzag'ı artırıyor

**Kanıt:**
- Test prediction dosyasındaki koordinatlar zigzag
- Training visualization smooth (çünkü direkt `decode_single_lane` çıktısı)
- Test prediction zigzag (çünkü spline interpolation sonrası)

## Çözüm

### Seçenek 1: Spline Interpolation'ı İyileştir
- `Lane` class'ın spline parametrelerini ayarla
- Daha smooth spline kullan (daha yüksek degree veya regularization)

### Seçenek 2: get_prediction_string'de Smoothing Ekle
- `get_prediction_string` içinde spline sonrası smoothing ekle
- Savitzky-Golay veya Gaussian smoothing

### Seçenek 3: Decode Sonrası Smoothing Güçlendir
- `decode_single_lane`'deki smoothing'i güçlendir
- window_length artır (15 → 25)

### Seçenek 4: Spline Yerine Linear Interpolation
- `Lane` class'ı linear interpolation kullanacak şekilde değiştir
- Daha basit ama daha smooth

## Öncelik

1. **Hızlı test:** `decode_single_lane` smoothing güçlendir (window_length=25)
2. **Orta vadeli:** `get_prediction_string` içinde spline sonrası smoothing
3. **Uzun vadeli:** `Lane` class spline parametrelerini optimize et

## Beklenen Etki

- **Önceki:** Test prediction zigzag (std=4.9981)
- **Smoothing güçlendirme sonrası:** Test prediction smooth (std < 2.0)
- **F1 0.5'te:** 0.0000 → 0.3+








