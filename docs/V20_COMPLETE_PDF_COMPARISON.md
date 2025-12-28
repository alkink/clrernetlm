# V20: PDF ile Tam Karşılaştırma - Root Cause Analizi

## Test Sonuçları (20251207_204436, 20251207_223016)

- **F1@0.5: ~0.02-0.03** (hala çok kötü)
- **"Şeritler yukarı yükseliyor"**: Devam ediyor
- **"Acayip zigzag oluştu"**: Devam ediyor (smoothing güçlendirildi ama yeterli değil)
- **Şeritlere oturmuyor**: Devam ediyor

## PDF ile Tam Karşılaştırma

### 1. Y Koordinatı - PDF'de Nasıl?

**PDF (Line 233-237):**
> "y coordinate is equally and vertically sampled at regression time step t, i.e. yt = H/T · t, where H is image height."

**PDF (Line 237-238):**
> "Specifically, the keypoint at time step t is composed of two tokens, i.e. (xt, t)."

**PDF (Line 246-247):**
> "It should be noted that 0 and T are padding tokens for x, y coordinates respectively (i.e. if there is no lane at time step t, let xt = 0 and yt = T)."

**PDF (Line 260, Eq. 2):**
> "et = Ey(yt) + Ex(xt) + PEkeypoint(t)"

**PDF (Line 401-407):**
> "In training phase, every embedding (except for the last embedding of each sequence) output from language decoder will go through x head and y head to predict the next token id of x and y as illustrated in Eq. 3. Considering that predicting ordered y sequence is a simple task, we use two decoupled classification heads, x head, y head, to predict the next token of x and y, respectively"

**PDF (Line 477, Eq. 11):**
> "maximize ∑_{t=1}^T (logP(xt|x<t, Xv) + logP(yt|y<t, Xv))"

**PDF (Line 489-493):**
> "Inference. We sample tokens from the model likelihood P(xt|x<t, Xv) and P(yt|y<t, Xv) using the argmax sampling as illustrated in Eq. 3. The same as language models, we apply the standard greedy search with fixed length and EOS (the End of Sequence token) stop criteria to generate xy tokens at the same time (i.e. we stop prediction when EOS token is predicted or the current sequence reaches the max length). After obtaining the discrete tokens, we de-quantize them to get continuous coordinates."

### Analiz:

1. **PDF'de y koordinatı sabit (yt = H/T · t):** Y koordinatı t'den hesaplanıyor, model predict etmiyor.
2. **Ama PDF'de y head var ve Y-loss var!** Bu çelişkili görünüyor.
3. **PDF'de inference'da y_tok sample ediliyor:** "We sample tokens from the model likelihood P(xt|x<t, Xv) and P(yt|y<t, Xv)"

**Çözüm:** PDF'de y koordinatı hem sabit (yt = H/T · t) hem de model predict ediyor (logP(yt|y<t, Xv)). Bu durumda:
- Training'de y_tok = t (sabit) kullanılmalı
- Model y_tok predict ediyor ama loss'ta y_tok = t ile karşılaştırılıyor (çünkü y_tok zaten t olmalı)
- Inference'da y_tok sample ediliyor ama decode'da kullanılmıyor (çünkü y koordinatı t'den hesaplanıyor)

**Bizim kodda:**
- Training'de y_tokens = t (sabit) kullanılıyor ✅
- Model y_tok predict ediyor ✅
- Y-loss açık ✅
- Inference'da y_fixed = t (sabit) kullanılıyor ✅
- Decode'da y = sample_ys[t] kullanılıyor (y_tok kullanılmıyor) ✅

**Sorun:** Decode'da y_tok kullanılmıyor, bu doğru. Ama "şeritler yukarı yükseliyor" sorunu devam ediyor. Bu, y koordinatı transform'unda veya sample_ys hesaplamasında sorun olabilir.

### 2. Sample_ys Hesaplaması

**Kod (tokenizer.py, line 48-51):**
```python
def _compute_sample_ys(self) -> np.ndarray:
    """Uniformly spaced y positions from top (0) to bottom (img_h)."""
    # Use linspace with endpoint=False to get T positions in [0, img_h)
    return np.linspace(0.0, float(self.cfg.img_h), num=self.cfg.num_steps, endpoint=False)
```

**PDF'de "yt = H/T · t":**
- t=0: y=0
- t=1: y=H/T = 320/40 = 8.0
- ...
- t=39: y=H/T · 39 = 8.0 · 39 = 312.0

**Bizim kodda (endpoint=False):**
- t=0: y=0.0
- t=1: y=8.0
- ...
- t=39: y=312.0

Bu uyumlu görünüyor. Ama "şeritler yukarı yükseliyor" sorunu devam ediyor.

### 3. Y Koordinatı Transform

**Kod (lanelm_detector.py, line 280-283):**
```python
y_scale = float(y_max - y_min) / float(img_h)  # (590-270)/320 = 1.0
y_orig = ys * y_scale + float(y_min)  # ys * 1.0 + 270 = ys + 270
y_norm = y_orig / float(ori_img_h)  # (ys + 270) / 590
```

**Sorun:** `y_scale = 1.0` çünkü crop height (320) = resized height (320). Bu doğru görünüyor.

Ama "şeritler yukarı yükseliyor" sorunu, y koordinatlarının yanlış transform edildiğini gösteriyor.

### 4. PDF vs Bizim - Tam Karşılaştırma

| Özellik | PDF | Bizim (V20) | Durum |
|---------|-----|-------------|-------|
| **FPN Levels** | P3+P4+P5 | Full FPN (V18) | ✅ Düzeltildi |
| **embed_dim** | 512 (LaneLM-512) | 512 (V17) | ✅ Düzeltildi |
| **num_layers** | 3 | 3 (V15) | ✅ Düzeltildi |
| **Y-Loss** | Var (Eq. 11) | Açık (V16) | ✅ Düzeltildi |
| **Y Koordinatı (Training)** | yt = H/T · t (sabit) | y_tokens = t (sabit) | ✅ Doğru |
| **Y Koordinatı (Inference)** | y_tok sample ediliyor | y_fixed = t (sabit) | ❓ **FARK VAR!** |
| **Y Koordinatı (Decode)** | ? | y = sample_ys[t] | ❓ **KONTROL ET!** |
| **Smoothing** | ? | Savitzky-Golay (window_length=31) | ✅ Güçlendirildi |
| **Batch Size** | 128 | 8 (overfit) / 1 | ❌ **FARK VAR!** |
| **Lq Noise Range** | -5 to +5 pixels | 5 pixels | ✅ Doğru |
| **Loss Computation** | Sadece Lgt | Sadece Lgt (V14) | ✅ Doğru |

### 5. KRİTİK FARK: Y Koordinatı Inference

**PDF (Line 489-493):**
> "Inference. We sample tokens from the model likelihood P(xt|x<t, Xv) and P(yt|y<t, Xv) using the argmax sampling"

PDF'de inference'da y_tok sample ediliyor!

**Bizim kodda (lanelm_detector.py, line 50):**
```python
y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).repeat(B, 1)
```

Bizim kodda y_fixed = t (sabit) kullanılıyor, y_tok sample edilmiyor!

**Bu çok kritik bir fark!** PDF'de y_tok sample ediliyor, bizde sabit kullanılıyor.

### 6. Olası Root Cause

1. **Y Koordinatı Inference Mismatch:** PDF'de y_tok sample ediliyor, bizde sabit. Bu training/inference mismatch yaratıyor olabilir.
2. **Y Koordinatı Decode:** Decode'da y_tok kullanılmıyor, bu doğru. Ama "şeritler yukarı yükseliyor" sorunu devam ediyor.
3. **Y Transform:** Y koordinatı transform'u yanlış olabilir.

## Sonraki Adımlar

1. **Y Koordinatı Inference'ı Düzelt:** PDF'de y_tok sample ediliyor, bizde sabit. Bu çok kritik!
2. **Y Koordinatı Decode Kontrol Et:** Decode'da y_tok kullanılıyor mu?
3. **Y Transform Kontrol Et:** Y koordinatı transform logic'ini düzelt






