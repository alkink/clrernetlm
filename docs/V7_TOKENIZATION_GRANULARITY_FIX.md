# V7: Tokenization Granularity Artırma - Kalıcı Çözüm

## Kök Neden: Kaba Quantization

### Mevcut Durum
- **nbins_x = 200**
- **img_w = 800**
- **Granularity: 800 / 200 = 4px per bin**

### Sorun
1. **Her token 4px'lik bir aralığı temsil ediyor**
   - Model 4px'lik adımlarla öğreniyor
   - Küçük hatalar (1-2px) birikiyor
   - Smoothness kaybı

2. **Zigzagging'in Ana Nedeni**
   - Model smooth pattern öğrenemiyor (çok kaba quantization)
   - Her adımda 4px'lik sıçramalar oluyor
   - Post-processing (smoothing, HR) yeterli değil

### PDF'de Ne Kullanılıyor?
- **nbins_x = 800** (Section 4.1, line 570)
- **Granularity: 800 / 800 = 1px per bin**
- Çok daha ince granularity → daha smooth predictions

## Kalıcı Çözüm: Granularity Artırma

### Değişiklik
- **nbins_x: 200 → 400** (2px per bin)
- **Veya: 200 → 800** (1px per bin, PDF standard)

### Neden Bu Kalıcı Çözüm?

1. **Model Seviyesinde:**
   - Post-processing değil, model architecture değişikliği
   - Model smoothness öğrenebilir
   - Küçük hatalar birikmez

2. **PDF'de Kanıtlanmış:**
   - PDF'de 800 bins kullanılıyor
   - Başarılı sonuçlar (F1@0.5 yüksek)

3. **Kök Nedeni Çözer:**
   - Kaba quantization → zigzagging
   - İnce granularity → smooth predictions

### Trade-off

**Avantajlar:**
- Daha smooth predictions
- Küçük hatalar birikmez
- Model seviyesinde çözüm
- PDF'de kanıtlanmış

**Dezavantajlar:**
- Model'i yeniden eğitmek gerekiyor
- Vocabulary size artıyor (200 → 400)
- Training biraz daha yavaş olabilir
- Memory kullanımı artabilir

## Uygulama

### 1. Training Script Güncellemesi

**Dosya:** `tools/train_lanelm_v4_fixed.py`
- `nbins_x = 400` (satır 211)

### 2. Test Config Güncellemesi

**Dosya:** `configs/lanelm/lanelm_v4_culane_test.py`
- `nbins_x = 400` (lanelm_cfg ve tokenizer_cfg)

### 3. Model Architecture

Model architecture otomatik olarak güncelleniyor:
- `LaneLMModel.__init__`: `nbins_x` parametresi
- `LaneLMHead.__init__`: `proj_x = nn.Linear(embed_dim, nbins_x)`
- `KeypointEmbedding.__init__`: `x_embedding = nn.Embedding(nbins_x, embed_dim)`

## Beklenen Etki

### Önceki Durum (200 bins)
- Granularity: 4px per bin
- Zigzagging: Yüksek
- F1@0.5: 0.05

### Sonraki Durum (400 bins)
- Granularity: 2px per bin
- Zigzagging: Orta-Düşük
- F1@0.5: 0.2-0.4

### PDF Standard (800 bins)
- Granularity: 1px per bin
- Zigzagging: Çok Düşük
- F1@0.5: 0.4-0.6

## Sonraki Adımlar

1. ✅ Training script güncellendi (nbins_x = 400)
2. ⏳ Test config güncelle (nbins_x = 400)
3. ⏳ Model'i yeniden eğit (400 bins ile)
4. ⏳ Test et ve sonuçları karşılaştır
5. ⏳ Gerekirse 800 bins'e çıkar (PDF standard)

## Not

Bu değişiklik mevcut checkpoint ile uyumlu değil. Model'i yeniden eğitmek gerekiyor. Ancak bu **en kalıcı çözüm** çünkü:
- Model seviyesinde smoothness sağlar
- PDF'de kanıtlanmış
- Kök nedeni çözer (kaba quantization)








