# V14: PDF ile Karşılaştırma Analizi

## Test Sonuçları

### Son Test (20251207_015806):
- **F1@0.5: 0.0465** (TP: 13, FP: 339, FN: 194) ⚠️ **HALA ÇOK KÖTÜ**
- **F1@0.1: 0.6512** (TP: 182, FP: 170, FN: 25) - Düşük threshold'da daha iyi ama yeterli değil

### Görsel Analiz:
- **Çok fazla hallucination**: Yol dışına çıkan çizgiler (kırmızı, sarı, macenta)
- **Çok fazla lane predict ediyor**: 5-8 çizgi görüyorum (max_lanes=4 olmasına rağmen!)
- **Yolları iyi coverlamıyor**: Şeritler gerçek yolları takip etmiyor
- **Geometrik bozukluk**: Zigzag, düzgün değil

## PDF vs Bizim Implementasyon - KRİTİK FARKLAR

### 1. Training Hyperparameters

**PDF (Sayfa 570):**
- Batch size: **128**
- nbins_x: **800** ✅ (Biz de 800 kullanıyoruz)
- Epochs: **100** ✅ (Biz de 100+ kullanıyoruz)
- Learning rate: **Belirtilmemiş** (muhtemelen 1e-3 veya 3e-4)

**Bizim:**
- Batch size: **1** (overfit test için) ⚠️ **FARKLI!**
- nbins_x: **800** ✅
- Epochs: **200+** (overfit test için)
- Learning rate: **3e-4** ✅

**Sorun:** Batch size=1 çok küçük! PDF'de 128 batch size kullanılıyor. Bu gradient variance'ı artırır ve training'i zorlaştırır.

### 2. Loss Computation - KRİTİK FARK!

**PDF (Sayfa 879-885, Eq. 10):**
- Loss **SADECE Lgt kısmında** hesaplanıyor
- Lq kısmında **LOSS YOK** (Lq sadece input, loss yok)
- PDF açıkça diyor: "Loss should only be computed on Lgt positions"

**Bizim:**
```python
# Lgt loss (main loss, weight 1.0)
loss_x_lgt = loss_x_fn(...)

# Lq loss (auxiliary loss, weight 0.5) - V13: Increased to learn Lq better
loss_x_lq = loss_x_fn(...)

# Combined loss: Lgt (1.0) + Lq (0.5)
loss_x = loss_x_lgt + 0.5 * loss_x_lq
```

**KRİTİK SORUN:** PDF'de Lq'da loss yok, bizde var! Bu PDF'deki stratejiye aykırı!

### 3. Presence Head - PDF'de YOK!

**PDF:**
- Presence head **bahsedilmiyor**
- PDF'de presence filtering yok
- Sadece HR (Hallucination Removal) algoritması var

**Bizim:**
- Presence head ekledik (V6)
- Presence loss weight: 0.5
- Presence filter threshold: 0.3

**Sorun:** PDF'de presence head yok, biz ekledik. Bu PDF'deki stratejiye aykırı olabilir.

### 4. Y-Loss - PDF'de Bahsedilmiyor

**PDF:**
- Y-loss **bahsedilmiyor**
- Sadece X-loss kullanılıyor gibi görünüyor

**Bizim:**
- Y-loss kapalı (use_y_loss = False) ✅
- Bu doğru, PDF'de de yok

### 5. Lq Noise Range

**PDF (Sayfa 870, Section 3.4):**
- Lq noise: **-5 to +5 pixels** random shift

**Bizim:**
- Lq noise: **-10 to +10 pixels** (V13'te artırdık)

**Sorun:** PDF'de -5 to +5, bizde -10 to +10. Bu çok fazla noise olabilir.

### 6. Training Strategy - "*" Version

**PDF (Sayfa 867-871):**
- CULane için "*" versiyonu: CLRNet Lq ◦ GT Lgt
- Bipartite matching: Start point distance
- Loss **SADECE Lgt'de**

**Bizim:**
- ✅ CLRNet Lq ◦ GT Lgt (doğru)
- ✅ Bipartite matching (doğru)
- ❌ Loss hem Lgt'de hem Lq'da (YANLIŞ!)

## Root Cause Analizi

### Ana Sorun: Loss Computation Yanlış!

PDF'de açıkça belirtiliyor: Loss **SADECE Lgt kısmında** hesaplanmalı. Lq sadece input, loss yok.

Bizim kodda:
```python
# YANLIŞ: Lq'da da loss var
loss_x = loss_x_lgt + 0.5 * loss_x_lq
```

PDF'de:
```python
# DOĞRU: Sadece Lgt'de loss
loss_x = loss_x_lgt  # Lq'da loss yok!
```

### İkinci Sorun: Batch Size Çok Küçük

PDF'de batch size=128, bizde batch size=1. Bu gradient variance'ı artırır ve training'i zorlaştırır.

### Üçüncü Sorun: Presence Head PDF'de Yok

PDF'de presence head yok, biz ekledik. Bu PDF'deki stratejiye aykırı olabilir.

## Çözüm Önerileri

### 1. Loss Computation'ı Düzelt (KRİTİK!)

```python
# PDF'ye göre: Loss SADECE Lgt'de
loss_x = loss_x_lgt  # Lq loss'u kaldır!
loss_y = loss_y_lgt  # Lq Y-loss'u kaldır!
```

### 2. Batch Size'ı Artır

```python
# Overfit test için bile batch_size=8-16 kullan
batch_size = 8  # Minimum, ideal: 32-64
```

### 3. Lq Noise'u Azalt

```python
# PDF'ye göre: -5 to +5 pixels
lq_noise_range = 5  # 10'dan 5'e düşür
```

### 4. Presence Head'i Kaldır veya Devre Dışı Bırak

```python
# PDF'de presence head yok, test için kaldır
presence_weight = 0.0  # Devre dışı bırak
```

## Sonraki Adımlar

1. ✅ PDF analizi tamamlandı
2. ⏳ Loss computation'ı düzelt (Lq loss'u kaldır)
3. ⏳ Batch size'ı artır (1 → 8-16)
4. ⏳ Lq noise'u azalt (10 → 5)
5. ⏳ Presence head'i devre dışı bırak (test için)
6. ⏳ Modeli yeniden eğit
7. ⏳ Test et ve sonuçları analiz et






