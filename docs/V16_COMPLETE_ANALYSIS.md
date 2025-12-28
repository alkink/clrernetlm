# V16: Tam Analiz - "Şeritler Yukarı Yükseliyor" Sorunu

## Test Sonuçları (20251207_140154)

- **F1@0.5: 0.0231** (TP: 7, FP: 393, FN: 200) ⚠️ **HALA ÇOK KÖTÜ**
- **F1@0.1: 0.4580** (TP: 139, FP: 261, FN: 68)

### Görsel Analiz - KRİTİK BULGULAR:

1. **"Şeritler yukarı yükseliyor"**: Yolun sonuna doğru şeritler yukarı doğru yükseliyor
2. **Şeritlere oturmuyor**: Şeritler gerçek şeritlere oturmuyor
3. **Çok fazla hallucination**: Yol dışına çıkan çizgiler
4. **Çok fazla lane predict ediyor**: max_lanes=4 olmasına rağmen 7-8 çizgi görülüyor

## Root Cause Analizi

### 1. Y Koordinatı Decode Sorunu (KRİTİK!)

**Kod (tokenizer.py, line 250):**
```python
y = sample_ys[t]  # t = step index [0, T-1]
```

**Sorun:** Decode'da `y_tok` (model prediction) kullanılmıyor, sadece `t` (step index) kullanılıyor!

**PDF (Line 233):**
> "y coordinate is equally and vertically sampled at regression time step t, i.e. yt = H/T · t"

PDF'de y koordinatı **sabit** (t'den hesaplanıyor), model predict etmiyor. Ama bizim kodda:
- Model y_tokens predict ediyor (logits_y var)
- Decode'da y_tok kullanılmıyor (sadece t kullanılıyor)

**Bu tutarlı görünüyor, ama kullanıcı "şeritler yukarı yükseliyor" diyor!**

### 2. Sample_ys Hesaplaması

**Kod (tokenizer.py, line 48-51):**
```python
def _compute_sample_ys(self) -> np.ndarray:
    """Uniformly spaced y positions from top (0) to bottom (img_h)."""
    # Use linspace with endpoint=False to get T positions in [0, img_h)
    return np.linspace(0.0, float(self.cfg.img_h), num=self.cfg.num_steps, endpoint=False)
```

**Örnek (img_h=320, T=40):**
- `sample_ys[0] = 0.0` (top)
- `sample_ys[1] = 8.0`
- ...
- `sample_ys[39] = 312.0` (bottom, endpoint=False olduğu için 320 değil)

**Sorun:** `endpoint=False` kullanılıyor, bu yüzden en alt y koordinatı `img_h-1` değil, `img_h - img_h/T` oluyor!

**PDF'de nasıl?** PDF'de "yt = H/T · t" diyor, yani:
- t=0: y=0
- t=1: y=H/T
- ...
- t=T-1: y=H/T · (T-1) = H - H/T

Bu `endpoint=False` ile uyumlu görünüyor.

### 3. Coordinate Transformation Sorunu

**Kod (lanelm_detector.py, line 280-283):**
```python
y_scale = float(y_max - y_min) / float(img_h)  # (590-270)/320 = 1.0
y_orig = ys * y_scale + float(y_min)  # ys * 1.0 + 270 = ys + 270
y_norm = y_orig / float(ori_img_h)  # (ys + 270) / 590
```

**Sorun:** `y_scale = 1.0` çünkü crop height (320) = resized height (320). Bu doğru görünüyor.

Ama "şeritler yukarı yükseliyor" sorunu, y koordinatlarının yanlış transform edildiğini gösteriyor.

### 4. Y-Loss Kapalı

**Kod (train_lanelm_v4_fixed.py, line 326):**
```python
use_y_loss = False  # FIX 8: Y-loss tamamen kapalı (test için)
```

Y-loss kapalı olduğu için model y koordinatlarını öğrenemiyor. Ama decode'da y_tok kullanılmadığı için bu sorun değil gibi görünüyor.

**Ama PDF'de Y-loss var! (Line 477, Eq. 11):**
> "maximize ∑_{t=1}^T (logP(xt|x<t, Xv) + logP(yt|y<t, Xv))"

PDF'de Y-loss var! Ama bizim kodda kapalı!

## Çözüm Önerileri

### 1. Y-Loss'u Aç (PDF'ye Göre)

PDF'de Y-loss var, bizim kodda kapalı. Y-loss'u açmalıyız:

```python
use_y_loss = True  # PDF'de var (Eq. 11)
y_loss_start_epoch = 1  # Hemen başla
```

### 2. Decode'da Y-Tok Kullan (Eğer Model Predict Ediyorsa)

Eğer model y_tokens predict ediyorsa, decode'da kullanılmalı:

```python
# Decode'da y_tok kullan
if y_tok < len(sample_ys):
    y = sample_ys[y_tok]
else:
    y = sample_ys[t]  # Fallback
```

Ama PDF'de y koordinatı sabit (t'den hesaplanıyor), bu yüzden bu gerekli olmayabilir.

### 3. Sample_ys Hesaplamasını Kontrol Et

`endpoint=False` doğru mu? PDF'de "yt = H/T · t" diyor, bu `endpoint=False` ile uyumlu.

### 4. Coordinate Transformation'ı Kontrol Et

Y koordinatlarının transform edilmesi doğru mu? "Şeritler yukarı yükseliyor" sorunu, y koordinatlarının yanlış transform edildiğini gösteriyor.

## PDF vs Bizim Implementasyon - Y Koordinatı

| Özellik | PDF | Bizim |
|---------|-----|-------|
| **Y Koordinatı** | Sabit (yt = H/T · t) | Sabit (sample_ys[t]) ✅ |
| **Y-Loss** | Var (Eq. 11) | Kapalı ❌ |
| **Decode'da Y-Tok** | Kullanılmıyor (sabit) | Kullanılmıyor (sabit) ✅ |
| **Sample_ys** | H/T · t | linspace(0, H, T, endpoint=False) ✅ |

**Sorun:** PDF'de Y-loss var, bizim kodda kapalı! Bu modelin y koordinatlarını öğrenememesine neden olabilir.

## Sonraki Adımlar

1. **Y-Loss'u Aç:** PDF'de var, bizim kodda kapalı. Y-loss'u açmalıyız.
2. **Test Et:** Y-loss açıldıktan sonra modeli yeniden eğit ve test et.
3. **Decode'ı Kontrol Et:** Y koordinatlarının decode edilmesi doğru mu?






