# V20: Y Koordinatı Inference Düzeltmesi

## Kritik Bulgu

**PDF (Line 489-493):**
> "Inference. We sample tokens from the model likelihood P(xt|x<t, Xv) and P(yt|y<t, Xv) using the argmax sampling"

PDF'de **y_tok sample ediliyor**, bizim kodda **y_fixed = t (sabit)** kullanılıyordu!

## Sorun

1. **Training'de:** y_tokens = t (sabit) kullanılıyor ✅
2. **Inference'da (ÖNCE):** y_fixed = t (sabit) kullanılıyordu ❌
3. **PDF'de Inference:** y_tok sample ediliyor ✅

Bu training/inference mismatch yaratıyor!

## Çözüm

### 1. Inference'da y_tok Sample Et

**Önceki Kod (lanelm_detector.py, line 50-61):**
```python
y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).repeat(B, 1)
```

**Yeni Kod:**
```python
# V20: PDF'de y_tok sample ediliyor (Line 489-493)
y_out = torch.zeros(B, T, dtype=torch.long, device=model_device)
# ... autoregressive decode ...
pred_y = torch.argmax(logits_y[:, t, :], dim=-1)
y_out[:, t] = pred_y
```

### 2. Decode'da y_tok Kullan

**Önceki Kod (tokenizer.py, line 250):**
```python
y = sample_ys[t]  # Use step index t
```

**Yeni Kod:**
```python
# V20: Use predicted y_tok instead of step index t
y_tok_clamped = min(max(0, y_tok), len(sample_ys) - 1)
y = sample_ys[y_tok_clamped]
```

## Değişiklikler

1. **lanelm_detector.py:**
   - `y_fixed` → `y_out` (autoregressive y prediction)
   - `y_in` eklendi (shift-right logic for y)
   - `logits_y` kullanılarak `pred_y` hesaplanıyor
   - `y_out[:, t] = pred_y` ile y_tok sample ediliyor

2. **tokenizer.py:**
   - `decode_single_lane`'de `y = sample_ys[t]` → `y = sample_ys[y_tok_clamped]`
   - y_tok kullanılarak y koordinatı decode ediliyor

## Beklenen Sonuç

1. **Training/Inference Uyumu:** Artık inference'da da y_tok sample ediliyor, training ile uyumlu
2. **"Şeritler Yukarı Yükseliyor" Sorunu:** y_tok kullanılarak decode edildiği için düzelmeli
3. **Zigzag Sorunu:** Smoothing zaten güçlendirildi, y_tok kullanımı ile daha da iyileşmeli

## Notlar

- PDF'de y koordinatı hem sabit (yt = H/T · t) hem de model predict ediyor (logP(yt|y<t, Xv))
- Training'de y_tokens = t (sabit) kullanılıyor, ama model y_tok predict ediyor
- Inference'da y_tok sample edilmeli (PDF'ye göre)
- Decode'da y_tok kullanılarak y koordinatı hesaplanmalı






