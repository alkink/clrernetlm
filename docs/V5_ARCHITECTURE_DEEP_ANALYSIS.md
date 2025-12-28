# LANELM V5 MİMARİ - DERİN ANALİZ VE TASARIM PLANI

## 📋 İÇİNDEKİLER
1. [Mevcut Mimari (V4) Analizi](#mevcut-mimari-v4-analizi)
2. [Kök Neden Analizi](#kök-neden-analizi)
3. [Yeni Mimari Gereksinimleri](#yeni-mimari-gereksinimleri)
4. [V5 Mimari Tasarımı](#v5-mimari-tasarımı)
5. [Uygulama Planı](#uygulama-planı)
6. [Test ve Doğrulama Stratejisi](#test-ve-doğrulama-stratejisi)

---

## 1. MEVCUT MİMARİ (V4) ANALİZİ

### 1.1 Mimari Bileşenleri

#### **KeypointEmbedding**
```python
# Mevcut: x_tokens + y_tokens + pos_embedding + lane_embedding
keypoint_emb = x_emb + y_emb + pos_emb + lane_emb * 10.0
```
**Sorunlar:**
- X token embedding'i çok güçlü (geçmiş X dizisi model'in ana girdisi)
- Y token embedding'i zayıf (zaten sabit 0..T-1)
- Lane embedding signal boosting (x10) var ama yeterli değil

#### **Decoder Layer (Self-Attention + Cross-Attention)**
```python
# Self-attention: Geçmiş X token'ları arasında bağımlılık
attn_out = self_attn(tgt, tgt, tgt, causal_mask)

# Cross-attention: Visual tokens üzerinde dikkat
attn_out = cross_attn(tgt, memory, memory)
```
**Sorunlar:**
- Self-attention geçmiş X token'lara çok bağımlı
- Cross-attention var ama görsel bilgi yeterince güçlü değil
- Causal mask geçmiş token'lara odaklanmayı zorunlu kılıyor

#### **Training vs Inference Mismatch**
```python
# Training (Teacher Forcing):
x_in[:, 0] = x_tokens[:, 0]  # GT'nin ilk değeri
x_in[:, 1:] = x_tokens[:, :-1]  # GT shifted

# Inference (Autoregressive):
x_in[:, 0] = 0  # Padding token
x_in[:, 1:] = pred_tokens[:, :-1]  # Model'in kendi tahminleri
```
**Sorunlar:**
- Training'de GT ile başlıyor, inference'da padding ile başlıyor
- Training'de her adımda GT görüyor, inference'da kendi hatalarını biriktiriyor
- Exposure bias: Model kendi hatalarını görmüyor

---

## 2. KÖK NEDEN ANALİZİ

### 2.1 Temel Sorun: "X Language Model" Paradigması

**Mevcut Model Mantığı:**
```
Geçmiş X Dizisi → Self-Attention → Cross-Attention (Visual) → Gelecek X
```

**Sorun:**
- Model, geçmiş X dizisini **birincil sinyal** olarak kullanıyor
- Görsel bilgi **ikincil sinyal** (cross-attention ile ekleniyor)
- Geçmiş X yanlışsa, görsel bilgi yeterince güçlü değil ki düzeltme yapsın

### 2.2 Cross-Attention Zayıflığı

**Mevcut Durum:**
- Cross-attention weights uniform'a yakın (0.99 uniformity score)
- Model görsel bilgiyi görmezden geliyor
- Visual tokens çok fazla (P5: 250, Full FPN: ~6500)

**Neden Zayıf:**
1. **Query (tgt) çok güçlü:** Geçmiş X token'ları zaten yeterli bilgi veriyor
2. **Key/Value (memory) çok zayıf:** Visual tokens spatial bilgiyi kaybediyor
3. **Attention mekanizması yetersiz:** Query'nin gücü Key/Value'yu eziliyor

### 2.3 Training-Inference Mismatch

**Teacher Forcing:**
- Her adımda GT X token'ı görüyor
- Model "mükemmel geçmiş" ile öğreniyor
- Loss düşük (0.27) ama gerçekçi değil

**Autoregressive:**
- Her adımda kendi tahminini görüyor
- Model "hatalı geçmiş" ile çalışıyor
- Hatalar birikiyor → zigzagging

**Parallel Decode (Deneme):**
- Tüm adımlar için padding (0) girdisi
- Model görsel bilgiyi kullanamıyor
- Constant prediction (mode collapse)

---

## 3. YENİ MİMARİ GEREKSİNİMLERİ

### 3.1 Temel Prensipler

1. **Görsel Bilgi Birincil Sinyal Olmalı**
   - Visual tokens, X token'lardan daha güçlü olmalı
   - Cross-attention yerine daha direkt bir mekanizma

2. **Geçmiş X Bağımlılığı Azaltılmalı**
   - Self-attention zayıflatılmalı veya kaldırılmalı
   - X token embedding'i azaltılmalı

3. **Training-Inference Uyumu**
   - Training ve inference aynı rejimde çalışmalı
   - Exposure bias ortadan kaldırılmalı

4. **Y Koordinatı Sabit Kalmalı**
   - Y token'ları zaten sabit (0..T-1)
   - Y-loss gereksiz, sadece X-loss yeterli

### 3.2 Mimari Değişiklik Stratejileri

#### **Strateji A: Visual-First Decoder**
- Cross-attention'ı güçlendir
- Self-attention'ı zayıflat veya kaldır
- Visual tokens'ı daha güçlü encode et

#### **Strateji B: Non-Autoregressive Decoder**
- Tüm X token'larını paralel tahmin et
- Geçmiş X bağımlılığını tamamen kaldır
- Sadece görsel bilgi + Y grid kullan

#### **Strateji C: Hybrid Approach**
- İlk birkaç token için visual-first
- Sonraki token'lar için autoregressive (ama zayıf)

---

## 4. V5 MİMARİ TASARIMI

### 4.1 Seçilen Strateji: **Visual-First Decoder (Strateji A)**

**Neden:**
- En az invazif (mevcut kodu minimal değiştirerek)
- En hızlı implement edilebilir
- En az risk (mevcut başarıları koruyarak)

### 4.2 Mimari Değişiklikleri

#### **4.2.1 Visual Token Encoder Güçlendirme**

**Mevcut:**
```python
# P5 Only: (B, 64, 10, 25) -> 250 tokens
# Full FPN: (B, 64, 20, 50) + (B, 64, 10, 25) + (B, 64, 5, 13) -> ~6500 tokens
```

**Yeni:**
```python
# P5 Only + Spatial Pooling:
# (B, 64, 10, 25) -> Adaptive Pooling -> (B, 64, 5, 13) -> 65 tokens
# Daha az token, daha güçlü spatial bilgi
```

**Değişiklikler:**
1. **Adaptive Spatial Pooling:** Visual tokens sayısını azalt (250 -> 65)
2. **Stronger Positional Encoding:** 2D PE'yi güçlendir
3. **Feature Normalization:** LayerNorm + Feature Scaling

#### **4.2.2 Keypoint Embedding Zayıflatma**

**Mevcut:**
```python
keypoint_emb = x_emb + y_emb + pos_emb + lane_emb * 10.0
```

**Yeni:**
```python
# X embedding'i zayıflat (geçmiş X'e daha az bağımlılık)
x_emb_scaled = x_emb * 0.3  # 1.0 -> 0.3
# Y ve pos embedding'i koru
# Lane embedding'i güçlendir (görsel bilgi ile birlikte)
lane_emb_scaled = lane_emb * 15.0  # 10.0 -> 15.0
keypoint_emb = x_emb_scaled + y_emb + pos_emb + lane_emb_scaled
```

**Değişiklikler:**
1. **X Embedding Scaling:** 1.0 -> 0.3 (geçmiş X'e daha az bağımlılık)
2. **Lane Embedding Boost:** 10.0 -> 15.0 (hangi lane'i tahmin ettiğini vurgula)

#### **4.2.3 Decoder Layer Yeniden Tasarımı**

**Mevcut:**
```python
# 1. Self-attention (causal, güçlü)
# 2. Cross-attention (visual, zayıf)
# 3. FFN
```

**Yeni:**
```python
# 1. Cross-attention FIRST (visual, güçlü)
# 2. Self-attention SECOND (causal, zayıf)
# 3. FFN
# 4. Visual-Query Fusion (yeni)
```

**Değişiklikler:**
1. **Sıra Değişikliği:** Cross-attention önce, self-attention sonra
2. **Self-Attention Zayıflatma:** Dropout artır (0.0 -> 0.2)
3. **Cross-Attention Güçlendirme:** Multi-head sayısını artır (8 -> 16)
4. **Visual-Query Fusion:** Cross-attention output'unu query'ye ekle (residual connection)

#### **4.2.4 Training Stratejisi**

**Mevcut:**
```python
# Teacher Forcing: x_in = GT shifted
# Scheduled Sampling: %20 oranında model tahmini kullan
```

**Yeni:**
```python
# Visual-First Training:
# 1. İlk epoch'larda: Pure Teacher Forcing (stabilite)
# 2. Sonraki epoch'larda: Scheduled Sampling (%30-50)
# 3. Son epoch'larda: AR Rollout Loss (kısa sequence, 5-10 step)
```

**Değişiklikler:**
1. **Scheduled Sampling Artışı:** %20 -> %30-50
2. **AR Rollout Loss Ekleme:** 5-10 step autoregressive loss ekle
3. **Progressive Training:** Aşamalı olarak exposure bias'ı azalt

---

## 5. UYGULAMA PLANI

### 5.1 Faz 1: Visual Token Encoder Güçlendirme

**Adımlar:**
1. `VisualTokenEncoder`'a adaptive pooling ekle
2. Token sayısını azalt (250 -> 65)
3. 2D PE'yi güçlendir
4. Test: Token sayısı ve spatial bilgi korunuyor mu?

**Dosyalar:**
- `libs/models/lanelm/model.py`: `VisualTokenEncoder` class

**Beklenen Sonuç:**
- Visual tokens sayısı azalır
- Spatial bilgi korunur
- Cross-attention daha etkili olur

### 5.2 Faz 2: Keypoint Embedding Zayıflatma

**Adımlar:**
1. `KeypointEmbedding`'e scaling parametreleri ekle
2. X embedding'i 0.3'e scale et
3. Lane embedding'i 15.0'a boost et
4. Test: Geçmiş X bağımlılığı azalıyor mu?

**Dosyalar:**
- `libs/models/lanelm/model.py`: `KeypointEmbedding` class

**Beklenen Sonuç:**
- Geçmiş X token'lara bağımlılık azalır
- Görsel bilgi daha önemli hale gelir
- Training loss biraz artabilir (normal)

### 5.3 Faz 3: Decoder Layer Yeniden Tasarımı

**Adımlar:**
1. `LaneLMDecoderLayer`'da sıra değişikliği (cross-attention önce)
2. Self-attention dropout artır (0.0 -> 0.2)
3. Cross-attention head sayısını artır (8 -> 16)
4. Visual-Query Fusion ekle
5. Test: Cross-attention weights daha non-uniform mu?

**Dosyalar:**
- `libs/models/lanelm/model.py`: `LaneLMDecoderLayer` class

**Beklenen Sonuç:**
- Cross-attention weights non-uniform olur
- Görsel bilgi daha etkili kullanılır
- Self-attention zayıflar

### 5.4 Faz 4: Training Stratejisi Güncelleme

**Adımlar:**
1. Scheduled Sampling oranını artır (%20 -> %30-50)
2. AR Rollout Loss ekle (5-10 step)
3. Progressive training schedule ekle
4. Test: Autoregressive inference hatası azalıyor mu?

**Dosyalar:**
- `tools/train_lanelm_v4_fixed.py`: Training loop

**Beklenen Sonuç:**
- Training ve inference rejimleri yakınlaşır
- Exposure bias azalır
- Autoregressive inference hatası azalır

### 5.5 Faz 5: Inference Optimizasyonu

**Adımlar:**
1. Inference'da visual-first decode kullan
2. Geçmiş X token'lara daha az bağımlılık
3. Smoothing güçlendir (window_length=15 -> 21)
4. Test: Zigzagging azalıyor mu?

**Dosyalar:**
- `libs/models/detectors/lanelm_detector.py`: `autoregressive_decode`
- `tools/train_lanelm_v4_fixed.py`: `visualize` function

**Beklenen Sonuç:**
- Zigzagging azalır
- Görsel kalite artar
- Inference hızı korunur

---

## 6. TEST VE DOĞRULAMA STRATEJİSİ

### 6.1 Her Faz İçin Test

**Faz 1 Test:**
- Visual token sayısı kontrolü
- Spatial bilgi korunuyor mu?
- Cross-attention uniformity score

**Faz 2 Test:**
- Geçmiş X bağımlılığı ölçümü
- Training loss değişimi
- Visual attention ağırlıkları

**Faz 3 Test:**
- Cross-attention weights non-uniform mu?
- Self-attention zayıfladı mı?
- Training loss stabil mi?

**Faz 4 Test:**
- Scheduled Sampling etkisi
- AR Rollout Loss etkisi
- Autoregressive inference hatası

**Faz 5 Test:**
- Zigzagging azaldı mı?
- Görsel kalite arttı mı?
- Inference hızı korundu mu?

### 6.2 Genel Test Senaryoları

**1-Image Overfit:**
- Loss < 0.3 olmalı
- Teacher Forcing: mean_err < 1px
- Autoregressive: mean_err < 20px (önceden 38px)

**8-Image Overfit:**
- Loss < 0.5 olmalı
- Görsel kalite iyi olmalı
- Zigzagging minimal olmalı

**100-Image Training:**
- Loss < 0.3 olmalı
- Görsel kalite iyi olmalı
- Zigzagging minimal olmalı

**Full Dataset Training:**
- Loss < 0.5 olmalı
- CULane F1 > 0.5 olmalı (önceden 0.0)

---

## 7. RİSK ANALİZİ VE YEDEK PLANLAR

### 7.1 Riskler

**Risk 1: Visual Token Sayısı Azaltma**
- **Risk:** Spatial bilgi kaybı
- **Yedek:** Adaptive pooling yerine learnable pooling kullan

**Risk 2: X Embedding Zayıflatma**
- **Risk:** Model hiç öğrenemez
- **Yedek:** Scaling'i daha yumuşak yap (0.3 -> 0.5)

**Risk 3: Decoder Sıra Değişikliği**
- **Risk:** Training instabil olur
- **Yedek:** Sıra değişikliği yerine fusion kullan

**Risk 4: Scheduled Sampling Artışı**
- **Risk:** Training yavaşlar
- **Yedek:** Progressive schedule kullan

### 7.2 Yedek Planlar

**Plan B: Non-Autoregressive Decoder**
- Eğer Strateji A başarısız olursa
- Tüm X token'larını paralel tahmin et
- Daha radikal ama daha etkili olabilir

**Plan C: Hybrid Approach**
- İlk 5 token visual-first
- Sonraki token'lar autoregressive
- En dengeli yaklaşım

---

## 8. BAŞARI KRİTERLERİ

### 8.1 Minimum Başarı Kriterleri

1. **1-Image Overfit:**
   - Loss < 0.3 ✅
   - Teacher Forcing: mean_err < 1px ✅
   - Autoregressive: mean_err < 25px (önceden 38px) ✅

2. **8-Image Overfit:**
   - Loss < 0.5 ✅
   - Görsel kalite iyi ✅
   - Zigzagging minimal ✅

3. **100-Image Training:**
   - Loss < 0.3 ✅
   - Görsel kalite iyi ✅
   - Zigzagging minimal ✅

### 8.2 İdeal Başarı Kriterleri

1. **Autoregressive Inference:**
   - mean_err < 15px (önceden 38px)
   - Zigzagging yok
   - Görsel kalite mükemmel

2. **Full Dataset Training:**
   - Loss < 0.4
   - CULane F1 > 0.6 (önceden 0.0)
   - Görsel kalite mükemmel

---

## 9. SONUÇ VE SONRAKİ ADIMLAR

### 9.1 Özet

**Mevcut Sorun:**
- Model "X language model" gibi çalışıyor
- Geçmiş X token'lara çok bağımlı
- Görsel bilgi yeterince güçlü değil
- Training-inference mismatch

**Çözüm:**
- Visual-First Decoder yaklaşımı
- Geçmiş X bağımlılığını azalt
- Görsel bilgiyi güçlendir
- Training-inference uyumunu artır

### 9.2 Sonraki Adımlar

1. **Faz 1'i uygula:** Visual Token Encoder güçlendirme
2. **Test et:** 1-image overfit ile doğrula
3. **Faz 2'yi uygula:** Keypoint Embedding zayıflatma
4. **Test et:** Geçmiş X bağımlılığı azaldı mı?
5. **Faz 3'ü uygula:** Decoder Layer yeniden tasarımı
6. **Test et:** Cross-attention weights non-uniform mu?
7. **Faz 4'ü uygula:** Training stratejisi güncelleme
8. **Test et:** Autoregressive inference hatası azaldı mı?
9. **Faz 5'i uygula:** Inference optimizasyonu
10. **Test et:** Zigzagging azaldı mı?

---

**Tarih:** 2024-12-30
**Versiyon:** 1.0
**Durum:** Analiz Tamamlandı, Uygulama Bekliyor








