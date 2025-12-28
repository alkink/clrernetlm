# V6: Presence Head Kalıcı Çözüm - Özet

## Sorun

Presence head her zaman yüksek skor veriyor (0.97-0.99) çünkü:
- Training'de: Negatif örnekler → sıfır embedding, pozitif örnekler → non-zero embedding
- Inference'da: Padding token'lar üretilse bile bazı valid token'lar var → non-zero embedding → presence head pozitif tahmin ediyor

## Kök Neden

Presence head "sıfır embedding = negatif, non-zero embedding = pozitif" öğrenmiş. Ama inference'da padding token'lar üretilse bile bazı valid token'lar olduğu için non-zero embedding oluşuyor.

## Kalıcı Çözüm

Presence head'in input'una `valid_ratio` (valid token sayısı / toplam token sayısı) eklendi:

1. **Pooling:** Valid token'ların hidden state'lerini pool et
2. **Valid Ratio:** Valid token sayısı / toplam token sayısı hesapla
3. **Concatenate:** Pooled embedding + valid_ratio → presence head input
4. **Presence Head:** Artık "valid_ratio ≈ 0 = negatif, valid_ratio > threshold = pozitif" öğrenebilir

## Yapılan Değişiklikler

### `libs/models/lanelm/model.py`

1. **Presence Head Architecture:**
   ```python
   # ÖNCE:
   nn.Linear(embed_dim, embed_dim)
   
   # SONRA:
   nn.Linear(embed_dim + 1, embed_dim)  # +1 for valid_ratio
   ```

2. **Presence Pooling:**
   ```python
   # ÖNCE:
   pooled = (hidden * valid_mask_f).sum(dim=1) / mask_sum_safe.float()
   presence_logits = self.presence_head(pooled)
   
   # SONRA:
   valid_count = valid_mask.sum(dim=1, keepdim=True).float()
   valid_ratio = valid_count / T
   pooled = (hidden * valid_mask_f).sum(dim=1) / mask_sum_safe
   pooled_with_ratio = torch.cat([pooled, valid_ratio], dim=1)
   presence_logits = self.presence_head(pooled_with_ratio)
   ```

## Sonraki Adımlar

1. ✅ Presence head architecture güncellendi
2. ✅ Presence pooling valid_ratio ekledi
3. ⏳ **Model'i yeniden eğitmek gerekiyor** (presence head architecture değişti, mevcut checkpoint uyumlu değil)
4. ⏳ Test script'ini çalıştır ve presence head'in doğru çalıştığını doğrula

## Not

Bu değişiklik mevcut checkpoint ile uyumlu değil. Model'i yeniden eğitmek gerekiyor. Training script'i (`train_lanelm_v4_fixed.py`) aynı kalıyor, sadece model architecture değişti.








