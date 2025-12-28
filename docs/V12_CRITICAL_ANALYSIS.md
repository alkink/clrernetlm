# V12: Kritik Analiz - Training/Test x_in Uyumsuzluğu

## Test Sonuçları (V12 "*" Version Sonrası)

### Test Sonuçları:
- **Test 1 (20251207_010822)**: F1@0.1: 0.3987, F1@0.5: 0.0000 ❌ (TP=0, FP=400, FN=207)
- **Test 2 (20251207_010230)**: F1@0.1: 0.4283, F1@0.5: 0.0560 ❌ (TP=17, FP=383, FN=190)

**Training Loss:** 0.0050 (çok iyi, overfitting) ✅
**Test F1@0.5:** 0.0000-0.0560 (çok kötü) ❌

**Sonuç:** Model training'de öğreniyor ama test'te çalışmıyor. **Temel sorun: Training/test x_in uyumsuzluğu!**

## Root Cause Analizi

### Sorun: Training/Test x_in Uyumsuzluğu (Kritik!)

**Training'de (train_lanelm_v4_fixed.py, line 632-634):**
```python
x_in_tf = x_tokens.clone()
x_in_tf[:, 1:] = x_tokens[:, :-1]  # Shift right: x_in[t] = x_tokens[t-1]
x_in_tf[:, 0] = x_tokens[:, 0]  # First token (Lq)
```

**Training mantığı:**
- t=0: x_in[:, 0] = x_tokens[:, 0] (Lq'nun ilk keypoint'i)
- t=1: x_in[:, 1] = x_tokens[:, 0] (Lq'nun ilk keypoint'i, çünkü shift right)
- t=2: x_in[:, 2] = x_tokens[:, 1] (Lq'nun ikinci keypoint'i)
- t=3: x_in[:, 3] = x_tokens[:, 2] (Lgt'nin ilk keypoint'i)

**Test'te (autoregressive_decode, line 72-80):**
```python
x_in = torch.zeros_like(x_out)
if t > 0:
    x_in[:, :t] = x_out[:, :t]  # Copy all previous tokens
    x_in[:, 0] = x_out[:, 0]  # Keep first token
```

**Test mantığı:**
- t=0: x_in[:, 0] = 0 (padding, çünkü t=0'da x_out henüz dolu değil)
- t=1: x_in[:, 0] = x_out[:, 0] (initial prompt), x_in[:, 1] = x_out[:, 1] (henüz predict edilmemiş!)
- t=2: x_in[:, 0] = x_out[:, 0], x_in[:, 1] = x_out[:, 1], x_in[:, 2] = x_out[:, 2] (henüz predict edilmemiş!)

**KRİTİK SORUN:** Test'te `x_in[:, t] = x_out[:, t]` yapıyoruz ama `x_out[:, t]` henüz predict edilmemiş! Training'de ise `x_in[:, t] = x_tokens[:, t-1]` yapıyoruz (shift right).

### PDF'den Bulgular

PDF'de training ve test'in aynı autoregressive mantığı kullanması gerekiyor:
- Training: Teacher forcing (x_in[t] = GT[t-1])
- Test: Autoregressive (x_in[t] = pred[t-1])

**Ama bizim kodumuzda:**
- Training: x_in[t] = x_tokens[t-1] (shift right) ✅
- Test: x_in[t] = x_out[t] (aynı timestep, yanlış!) ❌

## Çözüm: Test'te x_in'i Training ile Uyumlu Hale Getir

### Training Mantığı:
```python
x_in_tf[:, 1:] = x_tokens[:, :-1]  # Shift right
x_in_tf[:, 0] = x_tokens[:, 0]  # First token
```

### Test Mantığı (Düzeltilmiş):
```python
x_in = torch.zeros_like(x_out)
if t == 0:
    # t=0: Use initial prompt or padding
    if num_initial_points > 0:
        x_in[:, 0] = x_out[:, 0]  # Initial prompt
    else:
        x_in[:, 0] = pad_token_x  # Padding
else:
    # t>0: Shift right (same as training)
    x_in[:, :t] = x_out[:, :t-1]  # Shift right: x_in[t] = x_out[t-1]
    x_in[:, 0] = x_out[:, 0]  # Keep first token (training convention)
```

**KRİTİK:** `x_in[:, :t] = x_out[:, :t-1]` yapmalıyız, `x_out[:, :t]` değil!

## Alternatif: Training'i Test ile Uyumlu Hale Getir

Ama bu daha riskli çünkü training'i değiştirmek gerekiyor. Test'i düzeltmek daha güvenli.

## Notlar

- Training loss çok iyi (0.0050) ama test F1@0.5 = 0.0000
- Training/test x_in uyumsuzluğu kritik sorun
- Test'te `x_in[:, t] = x_out[:, t]` yanlış, `x_in[:, t] = x_out[:, t-1]` olmalı
- PDF'de autoregressive decoding shift right kullanıyor






