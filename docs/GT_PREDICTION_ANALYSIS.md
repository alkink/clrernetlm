# GT vs Prediction Analizi - Kritik Bulgular

## Debug Sonuçları Özeti

### 1. GT Loading (Original Space - 1640x590)
- **GT Lane 0:** X[-14.1, 732.8], Y[290.0, 510.0] ⚠️ **X negatif!**
- **GT Lane 1:** X[499.2, 774.8], Y[290.0, 590.0]
- **GT Lane 2:** X[815.4, 1409.6], Y[290.0, 590.0]
- **GT Lane 3:** X[862.6, 1650.4], Y[290.0, 440.0] ⚠️ **X > 1640!**

**Kritik:** GT'de image bounds dışı değerler var:
- Negatif X değerleri (-14.1)
- X > 1640 değerleri (1650.4)

### 2. Prediction (Normalized Space - 0-1)
- **Pred Lane 0:** X[0.0610, 0.4355], Y[0.4983, 0.8102]
- **Pred Lane 1:** X[0.3263, 0.4724], Y[0.4983, 0.9864]
- **Pred Lane 2:** X[0.5002, 0.8501], Y[0.4983, 0.9864]
- **Pred Lane 3:** X[0.5427, 0.9751], Y[0.4983, 0.7153]

**Kritik:** Tüm prediction'lar [0, 1) aralığında (normalized)

### 3. Prediction String (Original Space - 1640x590)
- **Pred String Lane 0:** X[88.1, 723.7], Y[290.0, 482.0] - 97 points
- **Pred String Lane 1:** X[535.1, 779.6], Y[290.0, 586.0] - 149 points
- **Pred String Lane 2:** X[811.2, 1400.7], Y[290.0, 586.0] - 149 points
- **Pred String Lane 3:** X[866.7, 1619.7], Y[290.0, 426.0] - 69 points

**Kritik:** Tüm prediction'lar image bounds içinde (X: 0-1640, Y: 290-590)

### 4. GT vs Prediction Comparison
- **Lane 0:** Center distance: 48.6 px, X overlap: True, Y overlap: True
- **Lane 1:** Center distance: 20.5 px, X overlap: True, Y overlap: True
- **Lane 2:** Center distance: 6.9 px, X overlap: True, Y overlap: True
- **Lane 3:** Center distance: 15.0 px, X overlap: True, Y overlap: True

**Kritik:** Tüm lane'ler için overlap var ve center distance'lar makul!

## Sorun Analizi

### Olası Sorun 1: GT Bounds Dışı Değerler

**GT'de image bounds dışı değerler var:**
- GT Lane 0: X=-14.1 (negatif)
- GT Lane 3: X=1650.4 (> 1640)

**Prediction'lar image bounds içinde:**
- Tüm X değerleri [0, 1640)
- Tüm Y değerleri [290, 590)

**CULaneMetric'in `draw_lane` fonksiyonu:**
- `draw_lane` fonksiyonu image bounds dışındaki değerleri nasıl handle ediyor?
- Eğer clip ediyorsa, GT ve prediction farklı şekilde clip edilebilir
- Bu, IoU hesaplamasını etkileyebilir

### Olası Sorun 2: Y Range Farkı

**GT Y range:**
- Lane 0: Y[290.0, 510.0] - 220 px range
- Lane 1: Y[290.0, 590.0] - 300 px range
- Lane 2: Y[290.0, 590.0] - 300 px range
- Lane 3: Y[290.0, 440.0] - 150 px range

**Prediction Y range:**
- Lane 0: Y[290.0, 482.0] - 192 px range (28 px eksik)
- Lane 1: Y[290.0, 586.0] - 296 px range (4 px eksik)
- Lane 2: Y[290.0, 586.0] - 296 px range (4 px eksik)
- Lane 3: Y[290.0, 426.0] - 136 px range (14 px eksik)

**Kritik:** Prediction'lar GT'den daha kısa (Y range eksik)

### Olası Sorun 3: Point Count Farkı

**GT point counts:**
- Lane 0: 23 points
- Lane 1: 31 points
- Lane 2: 31 points
- Lane 3: 16 points

**Prediction point counts:**
- Lane 0: 97 points (interpolated)
- Lane 1: 149 points (interpolated)
- Lane 2: 149 points (interpolated)
- Lane 3: 69 points (interpolated)

**Kritik:** Prediction'lar çok daha fazla point içeriyor (interpolated)

## Root Cause Hypothesis

### En Olası Sorun: GT Bounds Dışı Değerler + Y Range Eksikliği

1. **GT bounds dışı değerler:**
   - GT'de negatif X ve X > 1640 değerleri var
   - Prediction'lar image bounds içinde
   - `draw_lane` fonksiyonu bu değerleri farklı handle edebilir

2. **Y range eksikliği:**
   - Prediction'lar GT'den daha kısa (Y range eksik)
   - Bu, IoU hesaplamasını etkileyebilir (prediction GT'yi tam kapsamıyor)

3. **Point count farkı:**
   - Prediction'lar çok daha fazla point içeriyor (interpolated)
   - Bu normal (CULaneMetric interpolation yapıyor)

## Sonraki Adımlar

### 1. draw_lane Fonksiyonu Test
- `draw_lane` fonksiyonunu test et
- Image bounds dışındaki değerleri nasıl handle ediyor?
- GT ve prediction için aynı şekilde mi çalışıyor?

### 2. IoU Hesaplama Debug
- CULaneMetric'in IoU hesaplamasını debug et
- GT ve prediction'ın `draw_lane` çıktılarını karşılaştır
- IoU neden düşük?

### 3. Y Range Eksikliği Analizi
- Prediction'lar neden GT'den daha kısa?
- `get_prediction_string`'de Y range filtresi çok agresif mi?
- `lane_min_y` ve `lane_max_y` doğru mu?

## Öneriler

### 1. GT Bounds Dışı Değerleri Filtrele
- GT'deki image bounds dışı değerleri filtrele
- Veya prediction'ları da bounds dışına genişlet (ama bu mantıklı değil)

### 2. Y Range Filtresini Düzelt
- `get_prediction_string`'de Y range filtresini kontrol et
- `lane_min_y` ve `lane_max_y` doğru mu?
- Margin (0.01) yeterli mi?

### 3. draw_lane Fonksiyonunu İncele
- `draw_lane` fonksiyonunu detaylı incele
- Image bounds dışındaki değerleri nasıl handle ediyor?
- GT ve prediction için aynı şekilde mi çalışıyor?








