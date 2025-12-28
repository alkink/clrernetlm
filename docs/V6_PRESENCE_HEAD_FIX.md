# V6: Presence Head Kalıcı Çözüm

## Kök Neden Analizi

### Training'de Ne Oluyor?

1. **Negatif Örnekler (Padding Lanes):**
   - `x_tokens` ve `y_tokens` tamamen padding (satır 374-375)
   - `valid_mask` tamamen False
   - `mask_sum = 0` → `mask_sum_safe = 1`
   - `pooled = (hidden * 0).sum() / 1 = 0` (sıfır embedding)
   - Presence target = 0.0

2. **Pozitif Örnekler (GT Lanes):**
   - Valid token'lar var
   - `pooled` non-zero embedding
   - Presence target = 1.0

3. **Presence Head Öğrenmesi:**
   - "Sıfır embedding = negatif, non-zero embedding = pozitif" öğreniyor

### Inference'da Ne Oluyor?

1. **Model Autoregressive Decode:**
   - İlk timestep'lerde padding token (0) üretiyor
   - Sonra valid token'lar üretiyor
   - `valid_mask` bazı True değerler içeriyor

2. **Pooling:**
   - `pooled` non-zero embedding oluyor (valid token'lar var)
   - Presence head bunu pozitif olarak algılıyor → yüksek skor (0.97-0.99)

### Sorun

Presence head, "sıfır embedding = negatif" öğrenmiş ama inference'da padding token'lar üretilse bile bazı valid token'lar olduğu için non-zero embedding oluşuyor ve presence head bunu pozitif olarak algılıyor.

## Kalıcı Çözüm

### Valid Token Oranını Presence Head'e Eklemek

Presence head'in input'una `valid_ratio` (valid token sayısı / toplam token sayısı) ekliyoruz:

```python
valid_count = valid_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
T = x_tokens.shape[1]
valid_ratio = valid_count / T  # (B, 1) - ratio of valid tokens [0, 1]

pooled = (hidden * valid_mask_f).sum(dim=1) / mask_sum_safe  # (B, D)
pooled_with_ratio = torch.cat([pooled, valid_ratio], dim=1)  # (B, D+1)

presence_logits = self.presence_head(pooled_with_ratio)  # (B, 1)
```

### Neden Bu Çözüm?

1. **Training'de:**
   - Negatif örnekler: `valid_ratio = 0.0` → Presence head "valid_ratio ≈ 0 = negatif" öğrenir
   - Pozitif örnekler: `valid_ratio > 0.5` → Presence head "valid_ratio > threshold = pozitif" öğrenir

2. **Inference'da:**
   - Padding lane'ler: `valid_ratio ≈ 0` → Presence head negatif tahmin eder
   - Gerçek lane'ler: `valid_ratio > 0.5` → Presence head pozitif tahmin eder

3. **Avantajlar:**
   - Presence head artık valid token sayısına göre öğreniyor
   - Pooling embedding'i hala kullanılıyor (semantic bilgi)
   - Valid ratio ek bilgi sağlıyor (geometric bilgi)

## Yapılan Değişiklikler

1. **`libs/models/lanelm/model.py`:**
   - `presence_head` input dimension: `embed_dim` → `embed_dim + 1`
   - Presence pooling: `pooled` → `pooled_with_ratio` (valid_ratio concatenated)

## Sonraki Adımlar

1. ✅ Presence head input'una valid_ratio eklendi
2. ⏳ Model'i yeniden eğitmek gerekiyor (presence head architecture değişti)
3. ⏳ Test script'ini çalıştır ve presence head'in doğru çalıştığını doğrula

## Not

Bu değişiklik mevcut checkpoint ile uyumlu değil. Model'i yeniden eğitmek gerekiyor.
