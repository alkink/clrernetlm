# V15: PDF ile Tam Karşılaştırma - TÜM FARKLAR

## Test Sonuçları (20251207_015806)

- **F1@0.5: 0.0465** (TP: 13, FP: 339, FN: 194) ⚠️ **HALA ÇOK KÖTÜ**
- **F1@0.1: 0.6512** (TP: 182, FP: 170, FN: 25) - Düşük threshold'da daha iyi ama yeterli değil

### Görsel Analiz:
- **Şeritlere hiç oturmuyor**: Çizgiler gerçek şeritlerin üzerinde değil
- **Yolları iyi coverlamıyor**: Şeritler gerçek yolları takip etmiyor
- **Çok fazla hallucination**: Yol dışına çıkan çizgiler (kırmızı, sarı, macenta)
- **Geometrik bozukluk**: Zigzag, düzgün değil

## PDF vs Bizim Implementasyon - TÜM FARKLAR

### 1. ⚠️ KRİTİK: Decoder Layers Sayısı YANLIŞ!

**PDF (Line 382):**
> "This decoder consists of **3 layers** of LaneLM blocks."

**Bizim:**
```python
num_layers=4,  # V15: Kullanıcı 4'e çıkarmış, ama PDF'de 3!
```

**KRİTİK SORUN:** PDF'de açıkça **3 layers** yazıyor, bizde **4 layers** var! Bu model kapasitesini değiştirir ve PDF'deki sonuçlarla karşılaştırılamaz.

### 2. Loss Computation - Düzeltildi ✅

**PDF (Line 467-481, Eq. 10, 11):**
- Loss **SADECE Lgt kısmında** hesaplanıyor
- Lq kısmında **LOSS YOK** (Lq sadece input, loss yok)

**Bizim (V14 sonrası):**
```python
# ✅ DOĞRU: Sadece Lgt'de loss
loss_x = loss_x_lgt  # Lq loss'u kaldırıldı!
loss_y = loss_y_lgt  # Lq Y-loss'u kaldırıldı!
```

### 3. Batch Size - Düzeltildi ✅

**PDF (Line 570):**
- Batch size: **128**

**Bizim (V14 sonrası):**
```python
batch_size = 8 if args.overfit_size > 1 else 1  # Overfit test için minimum 8
```

**Not:** Overfit test için 8 yeterli, ama full training için 128 olmalı.

### 4. Lq Noise Range - Düzeltildi ✅

**PDF (Line 506, Section 3.4):**
- Lq noise: **-5 to +5 pixels** random shift

**Bizim (V14 sonrası):**
```python
lq_noise_range = 5  # PDF: "randomly shifting the x-coordinates by -5 to 5 pixels"
```

### 5. Presence Head - PDF'de YOK!

**PDF:**
- Presence head **bahsedilmiyor**
- PDF'de presence filtering yok
- Sadece HR (Hallucination Removal) algoritması var

**Bizim:**
- Presence head ekledik (V6)
- Presence loss weight: 0.3
- Presence filter threshold: 0.3

**Sorun:** PDF'de presence head yok, biz ekledik. Bu PDF'deki stratejiye aykırı olabilir.

### 6. Y-Loss - PDF'de Bahsedilmiyor

**PDF:**
- Y-loss **bahsedilmiyor**
- Sadece X-loss kullanılıyor gibi görünüyor

**Bizim:**
- Y-loss kapalı (use_y_loss = False) ✅
- Bu doğru, PDF'de de yok

### 7. Model Architecture - Diğer Hyperparameters

**PDF (Line 380-400):**
- Decoder: **3 layers** ⚠️ (Bizde 4!)
- embed_dim: **256** ✅ (Bizde 256)
- num_heads: **Belirtilmemiş** (muhtemelen 8)
- ffn_dim: **Belirtilmemiş** (muhtemelen 512)

**Bizim:**
```python
embed_dim=256,  # ✅ PDF ile uyumlu
num_layers=4,   # ❌ PDF'de 3!
num_heads=8,    # ✅ Muhtemelen doğru
ffn_dim=512,    # ✅ Muhtemelen doğru
```

### 8. Visual Encoder - PDF'de Detay Yok

**PDF (Line 344-375):**
- FPN kullanılıyor
- Multi-scale features: {F0, F1, F2}
- 2D positional embedding bahsedilmiyor (ama muhtemelen kullanılıyor)

**Bizim:**
- P5-only veya Full FPN (P3+P4+P5)
- 2D positional embedding kullanıyoruz ✅
- Adaptive pooling kullanıyoruz (PDF'de bahsedilmiyor)

### 9. Training Hyperparameters

**PDF (Line 570):**
- Batch size: **128**
- nbins_x: **800** ✅
- Epochs: **100** ✅
- Learning rate: **Belirtilmemiş** (muhtemelen 1e-3 veya 3e-4)

**Bizim:**
- Batch size: **8** (overfit test için) ⚠️
- nbins_x: **800** ✅
- Epochs: **200+** (overfit test için)
- Learning rate: **3e-4** ✅

### 10. Optimizer ve Scheduler

**PDF:**
- Optimizer: **Belirtilmemiş** (muhtemelen Adam)
- Scheduler: **Belirtilmemiş** (muhtemelen cosine annealing)

**Bizim:**
```python
optimizer = optim.Adam(lanelm.parameters(), lr=args.lr, weight_decay=0.0)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
```

### 11. Training Strategy - "*" Version

**PDF (Line 867-871):**
- CULane için "*" versiyonu: CLRNet Lq ◦ GT Lgt
- Bipartite matching: Start point distance
- Loss **SADECE Lgt'de**

**Bizim:**
- ✅ CLRNet Lq ◦ GT Lgt (doğru)
- ✅ Bipartite matching (doğru)
- ✅ Loss sadece Lgt'de (V14'te düzeltildi)

## Root Cause Analizi

### Ana Sorun: Decoder Layers Sayısı YANLIŞ!

PDF'de açıkça **3 layers** yazıyor, bizde **4 layers** var. Bu:
1. Model kapasitesini değiştirir
2. PDF'deki sonuçlarla karşılaştırılamaz
3. Overfitting riskini artırabilir
4. Training dinamiklerini değiştirir

### İkinci Sorun: Presence Head PDF'de Yok

PDF'de presence head bahsedilmiyor. Biz ekledik ama bu PDF'deki stratejiye aykırı olabilir.

### Üçüncü Sorun: Batch Size Çok Küçük (Overfit Test İçin)

Overfit test için batch_size=8 yeterli, ama full training için 128 olmalı.

## Çözüm Önerileri

### 1. ⚠️ KRİTİK: Decoder Layers'ı 3'e Düşür!

```python
num_layers=3,  # PDF'ye göre: "consists of 3 layers of LaneLM blocks" (line 382)
```

### 2. Presence Head'i Kaldır veya Devre Dışı Bırak

PDF'de presence head yok. Test için kullanıyoruz ama training'de presence loss'u kaldırabiliriz veya çok düşük weight ile kullanabiliriz.

### 3. Full Training İçin Batch Size'ı Artır

Overfit test için batch_size=8 yeterli, ama full training için 128 olmalı.

## PDF'den Önemli Notlar

1. **Line 1622-1623:** "LaneLM has learned the contextual representation of abrupt change points and consequently results in hallucination."
   - Bu, Lq ve Lgt arasındaki geçişte modelin "abrupt change points" öğrendiğini gösteriyor.
   - PDF'de bu sorun var, bizde de var.

2. **Line 879-880:** "*" versiyonunda LaneLM CLRNet'ten daha kötü çünkü CLRNet'in pseudo-label'larını predict ediyor.
   - Bu, "*" versiyonunun sınırlamasını gösteriyor.

3. **Line 382:** "This decoder consists of 3 layers of LaneLM blocks."
   - **KRİTİK:** PDF'de açıkça 3 layers yazıyor!

## Sonuç

En kritik fark: **Decoder layers sayısı yanlış!** PDF'de 3, bizde 4. Bu mutlaka düzeltilmeli.






