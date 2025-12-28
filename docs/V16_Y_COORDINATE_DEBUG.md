# V16: Y Koordinatı Decode Sorunu - "Şeritler Yukarı Yükseliyor"

## Test Sonuçları (20251207_140154)

- **F1@0.5: 0.0231** (TP: 7, FP: 393, FN: 200) ⚠️ **HALA ÇOK KÖTÜ**
- **F1@0.1: 0.4580** (TP: 139, FP: 261, FN: 68) - Düşük threshold'da daha iyi ama yeterli değil

### Görsel Analiz - KRİTİK BULGU:

Kullanıcının gözlemi: **"özellikle yolun sonuna doğru şeritler yukarı doğru yükseliyor ki böyle olmaması lazım"**

Bu çok kritik bir ipucu! Şeritlerin yukarı yükselmesi, y koordinatlarının yanlış decode edildiğini gösteriyor.

## Root Cause Analizi

### 1. Y Koordinatı Decode Sorunu

**Kod (tokenizer.py, line 250):**
```python
def decode_single_lane(self, x_tokens, y_tokens, smooth=False):
    sample_ys = self._compute_sample_ys()  # Uniformly spaced y positions
    ...
    for t in range(self.T):
        x_tok = int(x_tokens[t])
        y_tok = int(y_tokens[t])
        if x_tok == self.cfg.pad_token_x or y_tok >= self.T:
            continue
        
        # De-quantize x
        x = x_tok / max(1, self.cfg.nbins_x - 1) * (self.cfg.img_w - 1)
        y = sample_ys[t]  # <-- SORUN BURADA!
        xs.append(float(x))
        ys.append(float(y))
```

**Sorun:** `y = sample_ys[t]` kullanıyor, ama `y_tok` kullanmıyor! 

Yani decode sırasında y koordinatı her zaman `sample_ys[t]` (uniformly spaced) kullanılıyor, modelin predict ettiği `y_tok` kullanılmıyor!

### 2. PDF'de Y Koordinatı Nasıl Handle Ediliyor?

**PDF (Line 233-237):**
> "y coordinate is equally and vertically sampled at regression time step t, i.e. yt = H/T · t, where H is image height."

PDF'de y koordinatı **sabit** (t'den hesaplanıyor), model predict etmiyor. Ama bizim kodda model y_tokens predict ediyor ama decode'da kullanılmıyor!

### 3. Training'de Y-Loss Kapalı!

**Kod (train_lanelm_v4_fixed.py, line 326):**
```python
use_y_loss = False  # FIX 8: Y-loss tamamen kapalı (test için)
```

Y-loss kapalı olduğu için model y koordinatlarını öğrenemiyor! Ama decode'da y_tok kullanılmadığı için bu sorun değil gibi görünüyor.

### 4. Test'te Y-Fixed Kullanılıyor

**Kod (lanelm_detector.py, line 61):**
```python
y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).repeat(B, 1)
```

Test'te y_fixed = [0, 1, 2, ..., T-1] kullanılıyor, yani y_tokens = t (step index).

### 5. Decode'da Y-Tok Kullanılmıyor!

**Sorun:** Decode'da `y = sample_ys[t]` kullanılıyor, ama `y_tok` kullanılmıyor. 

Eğer model y_tok predict ediyorsa (ki ediyor, logits_y var), decode'da kullanılmalı!

## Çözüm Önerileri

### Seçenek 1: PDF'ye Göre Y Koordinatı Sabit (Önerilen)

PDF'de y koordinatı sabit (t'den hesaplanıyor). Bu durumda:
1. Model y_tokens predict etmemeli (sadece x_tokens)
2. Decode'da `y = sample_ys[t]` kullanılmalı (şu anki gibi)
3. Y-loss kapalı kalmalı (şu anki gibi)

**Ama sorun:** Şeritler yukarı yükseliyor! Bu, `sample_ys` hesaplamasında veya decode'da bir sorun olduğunu gösteriyor.

### Seçenek 2: Model Y Koordinatını Predict Etsin

Eğer model y_tokens predict ediyorsa, decode'da kullanılmalı:

```python
# Decode'da y_tok kullan
y = sample_ys[y_tok] if y_tok < len(sample_ys) else sample_ys[t]
```

Ama bu durumda Y-loss açık olmalı!

## Detaylı Analiz

### `_compute_sample_ys()` Fonksiyonu

**Kod (tokenizer.py, line 48-51):**
```python
def _compute_sample_ys(self) -> np.ndarray:
    """Uniformly spaced y positions from top (0) to bottom (img_h)."""
    # Use linspace with endpoint=False to get T positions in [0, img_h)
    return np.linspace(0.0, float(self.cfg.img_h), num=self.cfg.num_steps, endpoint=False)
```

Bu fonksiyon `[0, img_h)` aralığında T adet uniformly spaced y pozisyonu üretiyor.

**Örnek (img_h=320, T=40):**
- `sample_ys[0] = 0.0` (top)
- `sample_ys[1] = 8.0`
- `sample_ys[2] = 16.0`
- ...
- `sample_ys[39] = 312.0` (bottom, endpoint=False olduğu için 320 değil)

### Decode'da Y Kullanımı

**Kod (tokenizer.py, line 250):**
```python
y = sample_ys[t]  # t = step index [0, T-1]
```

Bu doğru görünüyor. Ama kullanıcı "şeritler yukarı yükseliyor" diyor. Bu, y koordinatlarının yanlış hesaplandığını gösteriyor.

### Olası Sorunlar

1. **Y-Loss Kapalı:** Model y koordinatlarını öğrenemiyor, ama decode'da kullanılmıyor zaten.
2. **Decode'da Y-Tok Kullanılmıyor:** Model y_tok predict ediyor ama decode'da kullanılmıyor.
3. **Sample_ys Hesaplaması Yanlış:** `endpoint=False` kullanılıyor, bu doğru mu?
4. **Coordinate Transformation:** Resized space'den original space'e dönüşümde sorun olabilir.

## Sonraki Adımlar

1. **Y-Loss'u Aç ve Test Et:** Model y koordinatlarını öğrensin, decode'da y_tok kullan.
2. **Decode'da Y-Tok Kullan:** `y = sample_ys[y_tok]` kullan, `y_tok` model prediction'ı.
3. **Sample_ys Hesaplamasını Kontrol Et:** `endpoint=False` doğru mu? PDF'de nasıl?

## PDF Referansı

**PDF (Line 233):**
> "y coordinate is equally and vertically sampled at regression time step t, i.e. yt = H/T · t"

PDF'de y koordinatı **sabit** (t'den hesaplanıyor), model predict etmiyor. Ama bizim kodda model y_tokens predict ediyor!

**PDF (Line 402):**
> "Considering that predicting ordered y sequence is a simple task, we use two decoupled classification heads, x head, y head, to predict the next token of x and y, respectively"

PDF'de y head var! Yani model y_tokens predict ediyor. Ama decode'da nasıl kullanılıyor?

**PDF (Line 477, Eq. 11):**
> "maximize ∑_{t=1}^T (logP(xt|x<t, Xv) + logP(yt|y<t, Xv))"

PDF'de Y-loss var! Ama bizim kodda kapalı!

## Kritik Soru

PDF'de y koordinatı hem sabit (yt = H/T · t) hem de model predict ediyor (logP(yt|y<t, Xv))? Bu çelişkili!

Muhtemelen:
- Training'de y koordinatı sabit (t'den hesaplanıyor)
- Model y_tokens predict ediyor ama loss'ta kullanılmıyor (çünkü zaten sabit)
- Decode'da y = sample_ys[t] kullanılıyor (sabit)

Ama kullanıcı "şeritler yukarı yükseliyor" diyor. Bu, decode'da bir sorun olduğunu gösteriyor.






