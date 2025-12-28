# V12: Kritik Fix - Model Forward ve x_in Uyumsuzluğu

## Sorun

**Test Sonuçları:**
- F1@0.5: 0.0000-0.0626 (çok kötü)
- Kullanıcı: "yolları da çok iyi coverlamıyor"

**Root Cause:**
1. **Model Forward Mantığı:** Model forward'da tüm sequence'i paralel işliyor, `logits_x` shape'i `(B, T, nbins_x)`. Test'te her timestep'te yeni bir forward yapıyoruz ama model tüm sequence'i görüyor.
2. **x_in Uyumsuzluğu:** Test'te her timestep'te yeni bir x_in oluşturuyoruz ama model forward'da tüm sequence'i paralel işliyor. Bu uyumsuzluk!
3. **PDF "Abrupt Change Points" Problemi:** Model Lq ve Lgt arasındaki "abrupt change points" öğreniyor ve bu hallucination'a yol açıyor (PDF sayfa 1619-1623).

## Analiz

### Training Mantığı:
```python
x_in_tf[:, 1:] = x_tokens[:, :-1]  # Shift right
x_in_tf[:, 0] = x_tokens[:, 0]  # First token
logits_x, _ = lanelm(vis_tok_batch, x_in_tf, y_in, ...)  # (B, T, nbins_x)
# Loss computed on logits_x[:, t, :] for all t
```

**Training'de:**
- x_in shape: (B, T)
- Model forward: Tüm sequence'i paralel işliyor
- logits_x shape: (B, T, nbins_x)
- Loss: Tüm timestep'ler için hesaplanıyor

### Test Mantığı (Şu Anki):
```python
for t in range(T):
    x_in = torch.zeros_like(x_out)  # (B, T)
    if t > 0:
        x_in[:, 1:t+1] = x_out[:, 0:t]  # Shift right
        x_in[:, 0] = x_out[:, 0]
    logits_x, _ = lanelm_model(visual_tokens, x_in, y_fixed, ...)  # (B, T, nbins_x)
    pred_x = torch.argmax(logits_x[:, t, :], dim=-1)  # Sadece t. timestep'i kullanıyoruz
    x_out[:, t] = pred_x
```

**Test'te:**
- Her timestep'te yeni bir x_in oluşturuyoruz
- Model forward: Tüm sequence'i paralel işliyor
- logits_x shape: (B, T, nbins_x)
- Sadece logits_x[:, t, :] kullanıyoruz

**Sorun:** Test'te her timestep'te yeni bir forward yapıyoruz ama model tüm sequence'i görüyor. x_in'in tüm pozisyonlarını doldurmalıyız!

### Çözüm: Test'te x_in'i Tüm Sequence İçin Doldur

**Test Mantığı (Düzeltilmiş):**
```python
for t in range(T):
    x_in = torch.zeros_like(x_out)  # (B, T)
    if t == 0:
        if num_initial_points > 0:
            x_in[:, 0] = x_out[:, 0]  # Initial prompt
        else:
            x_in[:, 0] = pad_token_x  # Padding
    else:
        # Shift right: x_in[t] = x_out[t-1] for all t
        x_in[:, 1:t+1] = x_out[:, 0:t]
        # Fill remaining positions with padding (model will use causal mask)
        x_in[:, t+1:] = pad_token_x
        # Keep first token (training convention)
        x_in[:, 0] = x_out[:, 0]
    
    logits_x, _ = lanelm_model(visual_tokens, x_in, y_fixed, ...)
    pred_x = torch.argmax(logits_x[:, t, :], dim=-1)
    x_out[:, t] = pred_x
```

**Ama bu da yeterli değil!** Model causal mask kullanıyor, yani t. timestep'te sadece [0, t-1] pozisyonlarını görebilir. Ama model forward'da tüm sequence'i paralel işliyor, bu yüzden x_in'in tüm pozisyonlarını doldurmalıyız.

**Daha İyi Çözüm:** Test'te x_in'i her timestep'te doğru şekilde doldur, ama model causal mask sayesinde sadece geçmiş timestep'leri görebilir.

## Notlar

- Model forward'da tüm sequence'i paralel işliyor
- Causal mask sayesinde model sadece geçmiş timestep'leri görebilir
- Test'te x_in'i her timestep'te doğru şekilde doldurmalıyız
- PDF'de "abrupt change points" problemi açıkça belirtilmiş






