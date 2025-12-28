# V12: x_in Fix - Training/Test Uyumsuzluğu Çözüldü

## Sorun

**Training'de (train_lanelm_v4_fixed.py, line 632-634):**
```python
x_in_tf = x_tokens.clone()
x_in_tf[:, 1:] = x_tokens[:, :-1]  # Shift right: x_in[t] = x_tokens[t-1]
x_in_tf[:, 0] = x_tokens[:, 0]  # First token
```

**Test'te (autoregressive_decode, önceki kod):**
```python
if t > 0:
    x_in[:, :t] = x_out[:, :t]  # Copy all previous tokens (YANLIŞ!)
    x_in[:, 0] = x_out[:, 0]
```

**Sorun:** Test'te `x_in[:, t] = x_out[:, t]` yapıyorduk ama `x_out[:, t]` henüz predict edilmemiş! Training'de ise `x_in[:, t] = x_tokens[:, t-1]` yapıyoruz (shift right).

## Çözüm

**Test'te (autoregressive_decode, yeni kod):**
```python
if t == 0:
    # t=0: Use initial prompt or padding
    if num_initial_points > 0:
        x_in[:, 0] = x_out[:, 0]  # Initial prompt
    else:
        x_in[:, 0] = pad_token_x  # Padding
else:
    # t>0: Shift right (same as training)
    # x_in[:, 1:t+1] = x_out[:, 0:t] means x_in[t] = x_out[t-1]
    x_in[:, 1:t+1] = x_out[:, 0:t]
    # Keep first token (training convention)
    x_in[:, 0] = x_out[:, 0]
```

**Mantık:**
- Training: x_in[t] = x_tokens[t-1] (shift right)
- Test: x_in[t] = x_out[t-1] (shift right, aynı mantık)

## Beklenen Etkiler

1. **Training/Test Uyumlu**
   - Training: x_in[t] = x_tokens[t-1] (shift right)
   - Test: x_in[t] = x_out[t-1] (shift right)
   - **Training/test uyumlu!**

2. **Model Doğru Autoregressive Decoding Yapar**
   - Test'te model önceki timestep'in prediction'ını kullanır
   - Training'de model önceki timestep'in GT'sini kullanır
   - **Aynı mantık, farklı veri kaynağı (GT vs pred)**

3. **F1@0.5 Skoru İyileşmeli**
   - Önceki: 0.0000-0.0560 (çok kötü)
   - Beklenen: >0.1 (minimum), ideal: >0.3
   - Training loss çok iyi (0.0050), test de iyileşmeli

## Notlar

- Training loss çok iyi (0.0050) ama test F1@0.5 = 0.0000
- Training/test x_in uyumsuzluğu kritik sorun
- Test'te shift right yapmalıyız, training ile aynı mantık
- PDF'de autoregressive decoding shift right kullanıyor






