# V17: Model Variants Analizi ve "Şeritler Yukarı Yükseliyor" Sorunu

## Test Sonuçları (20251207_181721)

- **F1@0.5: 0.0264** (TP: 8, FP: 392, FN: 199) ⚠️ **HALA ÇOK KÖTÜ**
- **F1@0.1: 0.3756** (TP: 114, FP: 286, FN: 93)

### Görsel Analiz - KRİTİK BULGULAR:

1. **"Şeritler yukarı yükseliyor"**: Yolun sonuna doğru şeritler yukarı doğru yükseliyor (devam ediyor!)
2. **Şeritlere oturmuyor**: Şeritler gerçek şeritlere oturmuyor
3. **Çok fazla hallucination**: Yol dışına çıkan çizgiler
4. **Çok fazla lane predict ediyor**: max_lanes=4 olmasına rağmen 7-8 çizgi görülüyor

## Model Variants - PDF vs Bizim

### PDF'de 3 Model Variant (Line 521-523):

| Variant | Encoder | embed_dim | Trainable Params |
|---------|---------|-----------|------------------|
| **LaneLM-128** | ResNet18 | 128 | 3.54MB |
| **LaneLM-256** | ResNet34 | 256 | 11.32MB |
| **LaneLM-512** | DLA34 | 512 | 39.65MB |

### CULane İçin Hangi Variant Kullanılıyor?

**PDF Table 3 (CULane Results):**
- LaneLM-128*: ResNet18 encoder
- LaneLM-256*: ResNet34 encoder
- LaneLM-512*: DLA34 encoder

**"*" versiyonu:** CLRNet'ten 2 keypoint prompt ile (PDF Line 867-868)

**En iyi sonuç:** LaneLM-512* (DLA34) - Total F1: 81.43

### Bizim Implementasyon:

| Özellik | PDF (LaneLM-512*) | Bizim | Durum |
|---------|-------------------|-------|-------|
| **Backbone** | DLA34 | DLA34 | ✅ |
| **embed_dim** | 512 | 256 | ❌ **YANLIŞ!** |
| **ffn_dim** | ? | 512 | ? |
| **num_layers** | 3 | 3 | ✅ |
| **num_heads** | ? | 8 | ? |
| **nbins_x** | 800 | 800 | ✅ |
| **num_steps (T)** | 40 | 40 | ✅ |

**KRİTİK SORUN:** `embed_dim=256` kullanıyoruz, ama PDF'de LaneLM-512 için `embed_dim=512` olmalı!

## "Şeritler Yukarı Yükseliyor" Sorunu - Detaylı Analiz

### 1. Y Koordinatı Decode

**Kod (tokenizer.py, line 250):**
```python
y = sample_ys[t]  # t = step index [0, T-1]
```

**PDF (Line 233):**
> "y coordinate is equally and vertically sampled at regression time step t, i.e. yt = H/T · t"

PDF'de y koordinatı **sabit** (t'den hesaplanıyor), model predict etmiyor. Decode'da `y_tok` kullanılmıyor, sadece `t` kullanılıyor. Bu doğru görünüyor.

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

**PDF'de "yt = H/T · t":**
- t=0: y=0
- t=1: y=H/T = 320/40 = 8.0
- ...
- t=39: y=H/T · 39 = 8.0 · 39 = 312.0

Bu `endpoint=False` ile uyumlu görünüyor.

### 3. Coordinate Transformation

**Kod (lanelm_detector.py, line 280-283):**
```python
y_scale = float(y_max - y_min) / float(img_h)  # (590-270)/320 = 1.0
y_orig = ys * y_scale + float(y_min)  # ys * 1.0 + 270 = ys + 270
y_norm = y_orig / float(ori_img_h)  # (ys + 270) / 590
```

**Sorun:** `y_scale = 1.0` çünkü crop height (320) = resized height (320). Bu doğru görünüyor.

Ama "şeritler yukarı yükseliyor" sorunu, y koordinatlarının yanlış transform edildiğini gösteriyor.

### 4. Olası Sorun: Y Koordinatı Ters mi?

Görsellerde şeritler yukarı yükseliyor. Bu, y koordinatlarının ters olduğunu gösterebilir!

**Normalde:**
- Y=0: Image top (uzak)
- Y=img_h: Image bottom (yakın)

**Ama belki:**
- Y=0: Image bottom (yakın)
- Y=img_h: Image top (uzak)

Bu durumda `sample_ys` hesaplaması ters olabilir!

## Çözüm Önerileri

### 1. embed_dim'ı 512'ye Çıkar (LaneLM-512)

PDF'de LaneLM-512 için `embed_dim=512` olmalı, bizim kodda `embed_dim=256`. Bu çok kritik!

### 2. Y Koordinatı Tersini Kontrol Et

"Şeritler yukarı yükseliyor" sorunu, y koordinatlarının ters olduğunu gösterebilir. `sample_ys` hesaplamasını kontrol etmeliyiz.

### 3. Coordinate Transformation'ı Kontrol Et

Y koordinatlarının transform edilmesi doğru mu? "Şeritler yukarı yükseliyor" sorunu, y koordinatlarının yanlış transform edildiğini gösteriyor.

## Sonraki Adımlar

1. **embed_dim'ı 512'ye Çıkar:** PDF'de LaneLM-512 için `embed_dim=512` olmalı
2. **Modeli Yeniden Eğit:** embed_dim=512 ile
3. **Test Et:** "Şeritler yukarı yükseliyor" sorunu düzelmiş mi?






