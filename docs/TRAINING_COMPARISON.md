# LaneLM Training Yaklaşımları Karşılaştırması

## 1. train_lanelm_v4_fixed.py (V4 - EN YENİ VE ÖNERİLEN)

### Özellikler:
- ✅ **Full FPN (P3+P4+P5)**: Daha fazla visual token (~5000), daha fazla spatial detay
- ✅ **2D Positional Embedding**: Spatial yapıyı korur, posterior collapse'ı önler
- ✅ **Absolute Tokenization**: Basit, öğrenmesi kolay (nbins_x=200)
- ✅ **LayerNorm**: FPN çıktılarını normalize eder
- ❌ **Y-Loss YOK**: Sadece X koordinatları öğreniliyor

### Kullanım:
- Overfit test için tasarlanmış (1-100 görüntü)
- Posterior collapse sorununu çözmüş
- En modern ve debug edilmiş versiyon

### Avantajlar:
- ✅ En iyi visual conditioning (Full FPN + 2D PE)
- ✅ Basit tokenization (absolute)
- ✅ Test edilmiş ve çalışıyor

### Dezavantajlar:
- ❌ Y-loss yok (Y koordinatları sabit)
- ❌ Explicit BOS token yok (lane_indices kullanılıyor)

---

## 2. train_lanelm_2k.py (2K Subset - Y-LOSS İLE)

### Özellikler:
- ⚠️ **P5 Only**: Daha az visual token (~200), daha yüksek semantic level
- ⚠️ **1D Positional Embedding**: Spatial yapıyı tam korumaz
- ⚠️ **Relative Disjoint Tokenization**: Daha karmaşık (nbins_x=200, max_abs_dx=32, vocab=300)
- ✅ **Explicit BOS Tokens (296-299)**: Lane ID'yi açıkça belirtir
- ✅ **Y-Loss VAR**: Hem X hem Y koordinatları öğreniliyor (0.5 weight each)

### Kullanım:
- 2000 görüntü subset için tasarlanmış
- Y koordinatlarını da öğrenmek için

### Avantajlar:
- ✅ Y-loss var (Y koordinatları öğreniliyor)
- ✅ Explicit BOS tokens (lane ID açık)
- ✅ 2K subset için test edilmiş

### Dezavantajlar:
- ❌ P5 only (daha az visual information)
- ❌ Relative tokenization (daha karmaşık)
- ❌ 1D PE (spatial yapıyı tam korumaz)

---

## 3. lanelm_culane_100imgs.py (Config - ESKİ VERSİYON)

### Özellikler:
- ⚠️ **P5 Only**: Daha az visual token
- ⚠️ **Relative Disjoint Tokenization**: Karmaşık
- ⚠️ **1D Positional Embedding**: Spatial yapıyı tam korumaz
- ✅ **Explicit BOS Tokens**: Lane ID açık

### Kullanım:
- Sadece config dosyası (test için)
- 100 görüntü test için

### Avantajlar:
- ✅ Explicit BOS tokens

### Dezavantajlar:
- ❌ Eski yaklaşım (P5 only, relative tokenization)
- ❌ Posterior collapse riski yüksek
- ❌ Test config'i, training script değil

---

## 🏆 ÖNERİ: train_lanelm_v4_fixed.py + Y-LOSS EKLE

### Neden V4 En İyi?

1. **Full FPN + 2D PE**: En iyi visual conditioning
   - Posterior collapse sorununu çözmüş
   - Spatial yapıyı korur

2. **Absolute Tokenization**: Basit ve etkili
   - Öğrenmesi kolay
   - Relative tokenization'dan daha stabil

3. **Test Edilmiş**: Overfit testlerde başarılı

### Eksik Olan: Y-LOSS

V4'e Y-loss ekleyerek en iyi yaklaşımı elde edebilirsiniz:

```python
# train_lanelm_v4_fixed.py'ye eklenebilir:
loss_y_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.T, reduction='mean')
loss_y = loss_y_fn(logits_y.view(B * T, -1), y_tokens.view(B * T))
loss = 0.5 * loss_x + 0.5 * loss_y  # X ve Y eşit ağırlık
```

### Sonuç:

**EN İYİ YAKLAŞIM: train_lanelm_v4_fixed.py + Y-LOSS**

- ✅ Full FPN + 2D PE (en iyi visual conditioning)
- ✅ Absolute tokenization (basit ve etkili)
- ✅ Y-loss (Y koordinatlarını da öğrenir)
- ✅ Test edilmiş ve çalışıyor

**ÖNERİLEN ADIMLAR:**

1. `train_lanelm_v4_fixed.py`'yi kullanın
2. Y-loss ekleyin (yukarıdaki kod)
3. Full dataset ile eğitim yapın (train_gt.txt)
4. Test edin (test.txt)








