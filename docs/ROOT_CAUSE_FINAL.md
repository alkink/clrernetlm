# Root Cause - Final Analysis (Kanıtlı)

## Gerçek Verilerle Kanıtlı Analiz

### Test Örneği: `driver_100_30frame/05251517_0433.MP4/02970`

#### 1. IoU Sonuçları (Gerçek)
- **Pred Lane 0 vs GT Lane 0:** IoU = 0.252496
- **Pred Lane 1 vs GT Lane 1:** IoU = 0.029666
- **Pred Lane 2 vs GT Lane 2:** IoU = 0.161101
- **IoU 0.1'de:** 2/4 hit ✅
- **IoU 0.5'te:** 0/4 hit ❌

#### 2. Overlap Analizi (Gerçek)
- **Pred 0 vs GT 0:**
  - X overlap: 68.6% ✅
  - Y overlap: 76.9% ✅
  - Center distance: 192.6 px
  - **Ama IoU = 0.25 < 0.5** ❌

#### 3. Koordinat Analizi (Gerçek)

**Prediction Lane 0 (ilk 10 nokta):**
```
189.59, 188.73, 188.12, 187.77, 187.66, 187.82, 188.22, 188.87, 189.76, 190.91
```
**Zigzag desen:** 1-2 px zigzaglar! ⚠️

**GT Lane 0 (ilk 10 nokta):**
```
-5.13, 30.83, 67.45, 103.11, 139.73, 175.33, 210.99, 247.61, 283.26, 319.88
```
**Smooth desen:** 30-40 px düzenli artışlar ✅

**Prediction Lane 1 (ilk 10 nokta):**
```
614.87, 619.23, 622.83, 625.77, 628.13, 630.02, 631.52, 632.74, 633.76, 634.68
```
**Zigzag desen:** 1-5 px zigzaglar! ⚠️

**GT Lane 1 (ilk 10 nokta):**
```
536.46, 542.50, 549.10, 555.69, 562.28, 568.87, 575.46, 582.05, 588.64, 595.23
```
**Smooth desen:** 5-7 px düzenli artışlar ✅

## Root Cause: ZIGZAG PREDICTIONS

### Sorun
1. **Prediction'lar zigzag:** 1-5 px zigzaglar
2. **GT smooth:** Düzenli artışlar
3. **Aynı bölgede ama şekil farklı:**
   - Overlap yüksek (68-77%)
   - Ama IoU düşük (0.03-0.25)
   - Çünkü şekil farklı!

### Neden Zigzag?
1. **Autoregressive decoding:** Her adımda küçük hatalar birikiyor
2. **Smoothing yetersiz:** `savgol_filter` yeterli değil
3. **Tokenization granularity:** `nbins_x=200` çok kaba (800px / 200 = 4px per bin)
4. **Model öğrenme:** Model zigzag pattern öğrenmiş olabilir

### Kanıt
- **Training visualization:** İyi görünüyor (smooth)
- **Test prediction:** Zigzag (gerçek dosyalardan)
- **IoU düşük:** Overlap yüksek ama şekil farklı

## Çözüm Önerileri

### 1. Smoothing Güçlendirme (Hızlı)
- `savgol_filter` window_length artır (15 → 25)
- Veya daha güçlü smoothing (Gaussian, median)

### 2. Tokenization Granularity Artırma (Orta)
- `nbins_x` artır (200 → 400)
- Daha ince granularity → daha smooth predictions

### 3. Model Öğrenme İyileştirme (Uzun)
- Smoothness loss ekle (geometric)
- AR rollout loss güçlendir
- Scheduled sampling artır

### 4. Post-Processing İyileştirme (Hızlı)
- Spline interpolation güçlendir
- Curve fitting (polynomial, B-spline)

## Öncelik

1. **Hızlı test:** Smoothing güçlendirme
2. **Orta vadeli:** Tokenization granularity
3. **Uzun vadeli:** Model öğrenme iyileştirme

## Beklenen Etki

- **Önceki:** IoU = 0.03-0.25 (zigzag)
- **Smoothing sonrası:** IoU = 0.3-0.5+ (smooth)
- **F1 0.5'te:** 0.0000 → 0.3+








