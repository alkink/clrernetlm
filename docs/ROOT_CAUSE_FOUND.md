# Root Cause Bulundu! 🎯

## Kritik Bulgular

### 1. ✅ Spline Interpolation: DOĞRU ÇALIŞIYOR
- Lane class spline interpolation doğru çalışıyor
- `get_prediction_string` mantığı doğru
- Normalization matematiksel olarak doğru

### 2. ⚠️ GT Bounds Dışı Değerler: SORUN!
- **GT Lane 0:** X=-14.1 (negatif) → IoU=0.0456 (düşük ama 0 değil)
- **GT Lane 3:** X=1650.4 (> 1640) → IoU=0.0000 ⚠️ **SIFIR!**
- **GT Lane 0 (clipped):** X[0.0, 732.8] → IoU=0.3150 (daha yüksek)

**Kritik:** GT'deki X > 1640 değerleri IoU'yu **sıfırlıyor**!

### 3. ⚠️ Y Range Eksikliği: SORUN!
- **GT Lane 0:** Y[290.0, 510.0] - 220 px range
- **Pred Lane 0:** Y[290.0, 482.0] - 192 px range (28 px eksik)
- **GT Lane 3:** Y[290.0, 440.0] - 150 px range
- **Pred Lane 3:** Y[290.0, 426.0] - 136 px range (14 px eksik)

**Kritik:** Prediction'lar GT'den daha kısa (Y range eksik)

### 4. ✅ Center Distance: MAKUL
- Lane 0: 48.6 px
- Lane 1: 20.5 px
- Lane 2: 6.9 px
- Lane 3: 15.0 px

**Kritik:** Center distance'lar makul, ama IoU düşük!

## Root Cause

### Ana Sorun: GT Bounds Dışı Değerler + Y Range Eksikliği

1. **GT X > 1640 değerleri:**
   - CULane dataset'inde normal (GT dosyalarında var)
   - Ama `draw_lane` fonksiyonu `cv2.line` kullanıyor
   - `cv2.line` image bounds dışındaki değerleri clip ediyor
   - Bu, GT ve prediction için farklı sonuçlar verebilir
   - **X > 1640 değerleri IoU'yu sıfırlıyor!**

2. **Y Range Eksikliği:**
   - Prediction'lar GT'den daha kısa
   - `get_prediction_string`'de Y range filtresi çok agresif olabilir
   - `lane_min_y` ve `lane_max_y` margin (0.01) yeterli olmayabilir
   - Bu, IoU'yu düşürüyor (prediction GT'yi tam kapsamıyor)

3. **Kombine Etki:**
   - GT bounds dışı değerler + Y range eksikliği = Çok düşük IoU
   - Test sonuçları: F1=0.0132 (neredeyse 0)

## Çözüm Önerileri

### 1. GT Bounds Dışı Değerleri Handle Et

**Seçenek A: GT'yi Clip Et (Önerilen)**
```python
# eval_predictions'da GT'yi clip et
for lane in anno:
    lane[:, 0] = np.clip(lane[:, 0], 0, 1639)
    lane[:, 1] = np.clip(lane[:, 1], 0, 589)
```

**Seçenek B: Prediction'ları Genişlet (Mantıklı Değil)**
- Prediction'ları X > 1640'a genişletmek mantıklı değil
- Model image bounds içinde tahmin yapıyor

**Seçenek C: draw_lane'de Clip Et**
- `draw_lane` fonksiyonunda clip et
- Ama bu GT'yi değiştirmek demek

### 2. Y Range Filtresini Düzelt

**Sorun:**
- `get_prediction_string`'de `lane_min_y` ve `lane_max_y` margin (0.01) çok küçük
- Bu, prediction'ların Y range'ini daraltıyor

**Çözüm:**
```python
# Margin'i artır
lane_min_y = lane.min_y - 0.05  # 0.01 → 0.05
lane_max_y = lane.max_y + 0.05  # 0.01 → 0.05
```

**Veya:**
- Y range filtresini kaldır
- Tüm Y değerleri için interpolasyon yap
- Invalid values'ı sonra filtrele

### 3. draw_lane Fonksiyonunu İyileştir

**Sorun:**
- `draw_lane` fonksiyonu image bounds kontrolü yapmıyor
- `cv2.line` OpenCV tarafından clip ediliyor
- Bu, GT ve prediction için farklı sonuçlar verebilir

**Çözüm:**
```python
def draw_lane(lane, img=None, img_shape=None, width=30, color=(255, 255, 255)):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    
    # Clip to image bounds BEFORE drawing
    h, w = img_shape[:2] if img_shape else img.shape[:2]
    lane_clipped = lane.copy()
    lane_clipped[:, 0] = np.clip(lane_clipped[:, 0], 0, w - 1)
    lane_clipped[:, 1] = np.clip(lane_clipped[:, 1], 0, h - 1)
    
    lane_clipped = lane_clipped.astype(np.int32)
    for p1, p2 in zip(lane_clipped[:-1], lane_clipped[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color, thickness=width)
    return img
```

## Test Stratejisi

### 1. GT Clipping Test
- GT'yi clip et ve test et
- IoU artıyor mu?

### 2. Y Range Margin Test
- Margin'i artır ve test et
- Y range eksikliği düzeliyor mu?

### 3. draw_lane Clip Test
- `draw_lane`'de clip et ve test et
- IoU artıyor mu?

## Sonraki Adımlar

1. ✅ **GT clipping test** - GT'yi clip et ve test et
2. ⏳ **Y range margin test** - Margin'i artır ve test et
3. ⏳ **draw_lane clip test** - `draw_lane`'de clip et ve test et
4. ⏳ **Kombine test** - Tüm düzeltmeleri birlikte test et

## Beklenen Sonuç

- GT clipping → IoU artışı (0.0000 → 0.3+)
- Y range margin artışı → IoU artışı (daha fazla overlap)
- Kombine → F1 artışı (0.0132 → 0.3+)








