# V19: Root Cause Analizi - "Şeritler Yukarı Yükseliyor" ve "Zigzag" Sorunları

## Test Sonuçları (20251207_190809, 20251207_202019)

- **F1@0.5: ~0.00-0.03** (hala çok kötü)
- **"Şeritler yukarı yükseliyor"**: Devam ediyor
- **"Acayip zigzag oluştu eskiden bu kadar değildi"**: YENİ SORUN!
- **Şeritlere oturmuyor**: Devam ediyor

## Root Cause Analizi

### 1. Y Koordinatı Decode Sorunu (KRİTİK!)

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
        y = sample_ys[t]  # <-- SORUN: y_tok kullanılmıyor, sadece t kullanılıyor!
        xs.append(float(x))
        ys.append(float(y))
```

**Sorun:** Decode'da `y_tok` (model prediction) kullanılmıyor, sadece `t` (step index) kullanılıyor!

**PDF'de Y Koordinatı (Line 233):**
> "y coordinate is equally and vertically sampled at fixed positions"

PDF'de y koordinatı **sabit** (t'den hesaplanıyor), ama model yine de `y_tok` predict ediyor. Bu bir **uyumsuzluk**!

**İki Olasılık:**
1. **PDF'de y_tok kullanılmıyor:** Y koordinatı sadece `t`'den hesaplanıyor, `y_tok` sadece padding/EOS kontrolü için.
2. **Bizim kodda yanlış:** `y_tok` kullanılmalı ama kullanılmıyor.

**Kontrol:** PDF'de y koordinatı nasıl decode ediliyor?

### 2. Zigzag Sorunu (YENİ!)

Kullanıcı: **"Acayip zigzag oluştu eskiden bu kadar değildi"**

**Olası Nedenler:**
1. **Smoothing kapalı:** `smooth=False` kullanılıyor olabilir
2. **Smoothing zayıf:** `window_length` çok küçük olabilir
3. **Full FPN'e geçiş:** Daha fazla visual token → daha fazla noise → zigzag
4. **Y-loss açılması:** Y-loss açılınca model y koordinatlarını öğrenmeye çalışıyor ama yanlış öğreniyor

**Kod (tokenizer.py, line 250-280):**
```python
def decode_single_lane(self, x_tokens, y_tokens, smooth=False):
    ...
    if smooth and len(xs) > 0:
        # Apply Savitzky-Golay filter
        window_length = min(25, len(xs) - 1 if len(xs) % 2 == 0 else len(xs))
        if window_length >= 3:
            xs = savgol_filter(xs, window_length, 3)
```

**Sorun:** `smooth=False` olabilir veya `window_length` çok küçük olabilir.

### 3. Y Koordinatı Transform Sorunu

**Kod (lanelm_detector.py, line 258-320):**
```python
def coords_to_lane_normalized(coords_resized, tokenizer_cfg, crop_bbox, img_w, img_h, ori_img_w, ori_img_h):
    xs = coords_resized[:, 0]
    ys = coords_resized[:, 1]
    x_min, y_min, x_max, y_max = crop_bbox
    
    # Clip to resized image bounds
    xs = np.clip(xs, 0.0, float(img_w - 1))
    ys = np.clip(ys, 0.0, float(img_h - 1))
    
    # Map resized x back to original-crop coordinates then normalize to [0,1)
    x_scale = float(ori_img_w) / float(img_w)
    y_scale = float(y_max - y_min) / float(img_h)
    
    x_orig = xs * x_scale
    y_orig = ys * y_scale + y_min  # <-- SORUN: y_min ekleniyor ama bu doğru mu?
```

**Sorun:** Y koordinatı transform'u yanlış olabilir. `y_min` ekleniyor ama bu crop bbox'un y_min'i, resized space'deki y koordinatını orijinal space'e map ederken yanlış olabilir.

### 4. PDF vs Bizim - Tam Karşılaştırma

| Özellik | PDF | Bizim (V19 Öncesi) | Durum |
|---------|-----|-------------------|-------|
| **FPN Levels** | P3+P4+P5 (3 levels) | P5 Only → Full FPN (V18) | ✅ Düzeltildi |
| **embed_dim** | 512 (LaneLM-512) | 256 → 512 (V17) | ✅ Düzeltildi |
| **num_layers** | 3 | 4 → 3 (V15) | ✅ Düzeltildi |
| **Y-Loss** | Var (Eq. 11) | Kapalı → Açık (V16) | ✅ Düzeltildi |
| **Y Koordinatı Decode** | ? (PDF'de belirtilmemiş) | `y = sample_ys[t]` (y_tok kullanılmıyor) | ❓ **KONTROL ET!** |
| **Smoothing** | ? (PDF'de belirtilmemiş) | Savitzky-Golay (window_length=25) | ❓ **KONTROL ET!** |
| **Y Transform** | ? (PDF'de belirtilmemiş) | `y_orig = ys * y_scale + y_min` | ❓ **KONTROL ET!** |

## Olası Root Cause'lar

### 1. Y Koordinatı Decode Sorunu (EN KRİTİK!)

**Hipotez:** PDF'de y koordinatı **sabit** (t'den hesaplanıyor), ama bizim kodda model `y_tok` predict ediyor ve bu kullanılmıyor. Bu bir **training/inference mismatch**!

**Çözüm:**
- PDF'de y koordinatı nasıl decode ediliyor kontrol et
- Eğer y koordinatı sabit ise, Y-loss'u kapat (çünkü model y_tok predict etmemeli)
- Eğer y koordinatı predict ediliyorsa, decode'da `y_tok` kullan

### 2. Zigzag Sorunu

**Hipotez:** Full FPN'e geçiş + Y-loss açılması → daha fazla noise → zigzag

**Çözüm:**
- Smoothing'i güçlendir (window_length artır)
- Smoothing'i her zaman aç (smooth=True)
- X koordinatı prediction'ını stabilize et

### 3. Y Transform Sorunu

**Hipotez:** Y koordinatı transform'u yanlış, bu yüzden "şeritler yukarı yükseliyor"

**Çözüm:**
- Y transform logic'ini kontrol et
- PDF'de y koordinatı nasıl transform ediliyor kontrol et

## Sonraki Adımlar

1. **PDF'de Y Koordinatı Decode Kontrol Et:** PDF'de y koordinatı nasıl decode ediliyor?
2. **Y-Loss'u Kapat (Test):** Eğer PDF'de y koordinatı sabit ise, Y-loss'u kapat
3. **Smoothing'i Güçlendir:** window_length artır, her zaman aç
4. **Y Transform Kontrol Et:** Y koordinatı transform logic'ini düzelt






