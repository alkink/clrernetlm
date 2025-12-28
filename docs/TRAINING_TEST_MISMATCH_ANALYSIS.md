# Training vs Test Mismatch Analizi

## Sorun Özeti
- **Training görselleştirmeleri:** Mükemmel görünüyor (şeritler GT'ye çok yakın)
- **Test sonuçları:** Çok kötü (IoU 0.5'te F1=0.0132, neredeyse 0)
- **Fark:** Training'de görselleştirmeler iyi ama test metrikleri kötü

## Test Sonuçları (100-image subset)

### Metrikler
- **IoU 0.1:** F1=0.3427 (kabul edilebilir ama ideal değil)
- **IoU 0.5:** F1=0.0132 ⚠️ **ÇOK KÖTÜ!**
- **IoU 0.75:** F1=0.0000 (neredeyse 0)
- **TP@0.5:** 4 (çok az true positive)
- **FP@0.5:** 396 (çok fazla false positive)
- **FN@0.5:** 203 (çok fazla false negative)

## Kritik Farklar: Training vs Test

### 1. **Koordinat Uzayı Farkı**

#### Training Visualization (`train_lanelm_v4_fixed.py::visualize`)
```python
# Resized space'de (800x320) görselleştirme
coords = tokenizer.decode_single_lane(x_tokens, y_tokens, smooth=True)
# coords: (N, 2) numpy array, X in [0, 800), Y in [0, 320)
# GT de resized space'de (800x320)
```

#### Test Inference (`LaneLMDetector.predict`)
```python
# 1. Resized space'de decode
coords_resized = self.tokenizer.decode_single_lane(x_tok[l], y_tok[l], smooth=True)
# coords_resized: (N, 2) numpy array, X in [0, 800), Y in [0, 320)

# 2. Normalized space'e çevir (0-1)
lane = coords_to_lane_normalized(
    coords_resized=coords_resized,
    tokenizer_cfg=self.tokenizer_cfg,
    crop_bbox=self.crop_bbox,
    img_w=800,
    img_h=320,
    ori_img_w=1640,
    ori_img_h=590,
)
# lane.points: (N, 2) normalized, X in [0, 1), Y in [0, 1)
```

**Sorun:** Training'de resized space'de görselleştirme yapılıyor, test'te normalized space'de değerlendirme yapılıyor. Bu conversion'da hata olabilir.

### 2. **Feature Extraction Farkı**

#### Training Visualization
```python
if use_p5_only:
    feats = extract_p5_feat(clrernet_model, imgs)  # Sadece P5
else:
    feats = extract_full_fpn_feats(clrernet_model, imgs)  # P3+P4+P5
```

#### Test Inference
```python
feats = self.extract_feat(imgs)  # LaneLMDetector.extract_feat
# visual_in_channels=(64,) → P5-only
# extract_feat P5-only için sadece son FPN level'ı döndürüyor
```

**Not:** Bu kısım aynı görünüyor (her ikisi de P5-only), ama `extract_p5_feat` vs `extract_feat` farklı implementasyonlar olabilir.

### 3. **Decode Mantığı**

#### Training Visualization
```python
all_preds = visual_first_decode(model, visual_tokens[:1], tokenizer, device, max_lanes)
# Returns: [(x_tokens, y_tokens), ...] for each lane
# x_tokens, y_tokens: numpy arrays
```

#### Test Inference
```python
x_tokens_all, y_tokens_all = autoregressive_decode(
    lanelm_model=self.lanelm.to(device),
    visual_tokens=visual_tokens,
    tokenizer_cfg=self.tokenizer_cfg,
    max_lanes=self.max_lanes,
    temperature=self.temperature,
)
# Returns: (B, max_lanes, T) tensors
```

**Not:** `visual_first_decode` ve `autoregressive_decode` aynı mantığı kullanmalı, ama return format'ı farklı.

### 4. **Normalization/Coordinate Conversion**

#### Training Visualization
- **Input:** Resized image (800x320)
- **Output:** Resized coordinates (800x320)
- **GT:** Resized coordinates (800x320)
- **Değerlendirme:** Görsel (gözle)

#### Test Inference
- **Input:** Resized image (800x320)
- **Output:** Normalized coordinates (0-1)
- **GT:** Original coordinates (1640x590) → Normalized (0-1)
- **Değerlendirme:** CULaneMetric (IoU hesaplama)

**Sorun:** `coords_to_lane_normalized` fonksiyonunda hata olabilir. Resized space'den normalized space'e çevirirken:
- Crop offset'i doğru uygulanıyor mu?
- Scale faktörleri doğru mu?
- Clipping doğru mu?

## Olası Nedenler

### 1. **Normalization Hatası (En Olası)**
`coords_to_lane_normalized` fonksiyonunda:
```python
x_scale = float(ori_img_w) / float(img_w)  # 1640 / 800 = 2.05
y_scale = float(y_max - y_min) / float(img_h)  # 320 / 320 = 1.0

x_orig = xs * x_scale  # Resized X → Original X
y_orig = ys * y_scale + float(y_min)  # Resized Y → Original Y (crop offset ekleniyor)

x_norm = x_orig / float(ori_img_w)  # Original X → Normalized X
y_norm = y_orig / float(ori_img_h)  # Original Y → Normalized Y
```

**Sorun:** Bu conversion doğru görünüyor ama test edilmeli. Özellikle:
- Crop offset (`y_min=270`) doğru uygulanıyor mu?
- Clipping `[0, 1)` aralığında mı?
- Spline interpolation doğru mu?

### 2. **Smoothing Farkı**
- Training'de `smooth=True` kullanılıyor
- Test'te de `smooth=True` kullanılıyor
- Ama smoothing parametreleri aynı mı? (window_length=15?)

### 3. **Hallucination Removal (HR)**
- Test'te HR uygulanıyor mu? (`LaneLMDetector`'da HR yok görünüyor)
- Training'de HR yok (görselleştirmede)
- HR çok agresifse, doğru tahminleri de silebilir

### 4. **GT Loading Farkı**
- Training'de GT resized space'de (pipeline'dan geliyor)
- Test'te GT original space'de (`.lines.txt` dosyasından)
- GT'nin test'te doğru yüklenip normalize edildiğinden emin olmalıyız

## Debug Stratejisi

### 1. **Normalization Doğrulama**
```python
# Test: Resized space'de bir koordinat al
x_resized = 400  # 800'in ortası
y_resized = 160  # 320'nin ortası

# Normalize et
x_orig = x_resized * (1640 / 800)  # = 820
y_orig = y_resized * (320 / 320) + 270  # = 430

x_norm = x_orig / 1640  # = 0.5
y_norm = y_orig / 590  # = 0.7288

# Bu değerler mantıklı mı?
```

### 2. **Training ve Test'te Aynı Görüntüyü Karşılaştır**
- Training'de görselleştirilen bir görüntüyü test'te de çalıştır
- Resized space'deki koordinatları karşılaştır
- Normalized space'deki koordinatları karşılaştır

### 3. **GT vs Prediction Overlay**
- Test'te GT ve prediction'ı aynı normalized space'de overlay yap
- Görsel olarak hizalı mı kontrol et

### 4. **IoU Hesaplama Debug**
- CULaneMetric'in IoU hesaplamasını debug et
- Prediction ve GT'nin normalized space'de doğru olduğundan emin ol

## Öneriler

### 1. **Normalization Fonksiyonunu Test Et**
- `coords_to_lane_normalized` fonksiyonunu unit test ile doğrula
- Bilinen resized koordinatlar için normalized koordinatları hesapla
- Beklenen değerlerle karşılaştır

### 2. **Training Visualization'ı Test Formatına Çevir**
- Training'deki `visualize` fonksiyonunu güncelle
- Normalized space'de görselleştirme yap
- GT'yi de normalized space'de yükle
- Bu şekilde test ile birebir aynı format'ta görselleştirme yap

### 3. **Test'te Resized Space Görselleştirme Ekle**
- Test'te prediction'ları resized space'de görselleştir
- Training görselleştirmeleriyle karşılaştır
- Fark varsa, decode mantığında sorun var demektir

### 4. **CULaneMetric Debug**
- CULaneMetric'in GT ve prediction loading'ini debug et
- Normalized space'deki koordinatları logla
- IoU hesaplamasını adım adım debug et

## Sonraki Adımlar

1. ✅ **Normalization fonksiyonunu test et**
2. ⏳ **Training visualization'ı test formatına çevir**
3. ⏳ **Test'te resized space görselleştirme ekle**
4. ⏳ **CULaneMetric debug**
5. ⏳ **Sorunları çöz**








