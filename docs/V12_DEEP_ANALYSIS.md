# V12: Derin Analiz - Training/Test Uyumsuzluğu ve Hallucination

## Test Sonuçları (x_in Fix Sonrası)

### Test Sonuçları:
- **Test 1 (20251207_011825)**: F1@0.1: 0.4415, F1@0.5: 0.0626 ❌ (TP=19, FP=381, FN=188)
- **Test 2 (20251207_012407)**: F1@0.1: 0.3822, F1@0.5: 0.0000 ❌ (TP=0, FP=400, FN=207)

**Training Loss:** 0.0050 (çok iyi, overfitting) ✅
**Test F1@0.5:** 0.0000-0.0626 (çok kötü) ❌

**Kullanıcı Feedback:** "yolları da çok iyi coverlamıyor" - Geometrik olarak da yanlış tahmin ediyor.

**Sonuç:** x_in fix'i yaptık ama hala sorun var. **Temel sorun: Model Lq ve Lgt arasındaki "abrupt change points" öğreniyor ve bu hallucination'a yol açıyor!**

## Root Cause Analizi

### Sorun 1: Model Forward Mantığı

**Model Forward (model.py, line 584-655):**
```python
def forward(self, visual_tokens, x_tokens, y_tokens, ...):
    # x_tokens: (B, T) - input sequence
    # Model processes ALL timesteps at once
    keypoint_emb = self.keypoint_embed(x_tokens, y_tokens, lane_indices)
    hidden = self.decoder(tgt=keypoint_emb, memory=visual_tokens, ...)
    logits_x, logits_y = self.head(hidden)  # (B, T, nbins_x), (B, T, max_y_tokens)
    return logits_x, logits_y
```

**Model tüm timestep'leri paralel işliyor!** Causal mask var ama model hala tüm sequence'i görüyor.

**Training'de (train_lanelm_v4_fixed.py, line 632-634):**
```python
x_in_tf[:, 1:] = x_tokens[:, :-1]  # Shift right: x_in[t] = x_tokens[t-1]
x_in_tf[:, 0] = x_tokens[:, 0]  # First token
```

**Training mantığı:**
- x_in[0] = x_tokens[0] (Lq'nun ilk keypoint'i)
- x_in[1] = x_tokens[0] (Lq'nun ilk keypoint'i, çünkü shift right)
- x_in[2] = x_tokens[1] (Lq'nun ikinci keypoint'i)
- x_in[3] = x_tokens[2] (Lgt'nin ilk keypoint'i - **ABRUPT CHANGE!**)

**Sorun:** Model x_in[3]'te Lgt'nin ilk keypoint'ini görüyor ama x_tokens[2]'de Lq'nun ikinci keypoint'i var. Bu "abrupt change" model'e öğretiliyor!

### Sorun 2: PDF'den Kritik Bulgu (Sayfa 1619-1623)

> "Analysis on hallucination. Current large language models are still struggling with hallucination.
> Figure 6(a) shows hallucination in LaneLM. Eq. 10 endows the model with the capability of VQA but
> it makes it easier for the model to predict cyclic sequences. **Figure 6(a) illustrates that the model has
> learned the abrupt change points that connecting Lq and Lgt on the side. LaneLM has learned the
> contextual representation of abrupt change points and consequently results in hallucination.**"

**KRİTİK:** Model Lq ve Lgt arasındaki "abrupt change points" öğreniyor ve bu hallucination'a yol açıyor!

### Sorun 3: Training/Test x_in Uyumsuzluğu (Devam Ediyor)

**Test'te (autoregressive_decode, yeni kod):**
```python
if t == 0:
    x_in[:, 0] = x_out[:, 0] if num_initial_points > 0 else pad_token_x
else:
    x_in[:, 1:t+1] = x_out[:, 0:t]  # Shift right: x_in[t] = x_out[t-1]
    x_in[:, 0] = x_out[:, 0]
```

**Test mantığı:**
- t=0: x_in[0] = x_out[0] (initial prompt)
- t=1: x_in[0] = x_out[0], x_in[1] = x_out[0] (initial prompt)
- t=2: x_in[0] = x_out[0], x_in[1] = x_out[0], x_in[2] = x_out[1] (initial prompt)
- t=3: x_in[0] = x_out[0], x_in[1] = x_out[0], x_in[2] = x_out[1], x_in[3] = x_out[2] (prediction)

**Ama model forward'da tüm timestep'leri paralel işliyor!** Test'te t=3'te x_in[3] = x_out[2] yapıyoruz ama model forward'da x_in'in tüm pozisyonlarını görüyor.

**KRİTİK SORUN:** Test'te her timestep'te yeni bir x_in oluşturuyoruz ama model forward'da tüm sequence'i paralel işliyor. Bu uyumsuzluk!

## Çözüm: Test'te Model Forward Mantığını Training ile Uyumlu Hale Getir

### Training Mantığı:
- Model forward'da tüm sequence'i paralel işliyor
- x_in[t] = x_tokens[t-1] (shift right)
- Model logits_x[:, t, :] üretiyor, bu t. timestep için prediction

### Test Mantığı (Düzeltilmiş):
- Her timestep'te yeni bir forward yapmalıyız
- Ama model forward'da tüm sequence'i paralel işliyor
- **Çözüm:** Test'te her timestep'te sadece o timestep için forward yapmalıyız, ama model tüm sequence'i görmeli

**Ama bu çok yavaş olur!** Alternatif: Test'te de training gibi tüm sequence'i bir kerede forward edelim, ama x_in'i doğru oluşturalım.

### Alternatif Çözüm: Test'te Parallel Decode Kullan

PDF'de parallel decode var mı? Kontrol etmeliyim.

## Notlar

- Training loss çok iyi (0.0050) ama test F1@0.5 = 0.0000-0.0626
- Model Lq ve Lgt arasındaki "abrupt change points" öğreniyor
- Training/test x_in uyumsuzluğu devam ediyor
- Model forward'da tüm sequence'i paralel işliyor, test'te ise timestep-by-timestep
- PDF'de "abrupt change points" problemi açıkça belirtilmiş






