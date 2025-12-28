# V7: Zigzagging Detaylı Analiz ve Kalıcı Çözüm

## Test Sonuçları (20251205_222239)

### Metrikler
- **F1@0.5 = 0.0165** ❌ (Çok düşük, önceki: 0.0494)
- **F1@0.1 = 0.6161** (Orta)
- **FP@0.5 = 395** (Çok yüksek)
- **TP@0.5 = 5** (Çok düşük)

### Görsel Analiz
- **Zigzagging:** Hala devam ediyor (tüm görsellerde açıkça görülüyor)
- **HR Algoritması:** Uygulanmış ama yeterli değil
- **Smoothing:** Uygulanmış ama yeterli değil

## Zigzagging'in Kök Nedenleri (PDF'ye Göre)

### 1. Tokenization Granularity Çok Kaba ⭐ EN KRİTİK

**Mevcut Durum:**
- `nbins_x = 200`
- `img_w = 800`
- **Granularity: 800 / 200 = 4px per bin**

**PDF'de Ne Kullanılıyor?**
- **Line 570:** "800 nbins and 100 training epochs"
- **Line 1641:** "nbins=800 aligns with our intuition"
- **Table 7:** `800 / 1280` → F1=97.64 (en iyi sonuç)
- **Granularity: 800 / 800 = 1px per bin** (CULane için)

**Sorun:**
- Her token 4px'lik bir aralığı temsil ediyor
- Model smooth pattern öğrenemiyor (çok kaba quantization)
- Küçük hatalar (1-2px) birikiyor → zigzagging

### 2. Prompting Strategy Eksik ⭐ PDF'DE KRİTİK

**PDF'de Ne Diyor? (Line 497-499):**
> "A regression network is employed to provide the two initial keypoints, for each lane. LaneLM is responsible for completing the remaining keypoints. The regression network (we use CLRNet [6]) only gives start points for each lane rather than the holistic lane."

**Mevcut Durum:**
- CLRNet'ten ilk 2 keypoint alınmıyor
- Model sıfırdan başlıyor (padding token ile)
- Bu, model'in başlangıç belirsizliğini artırıyor

**PDF'deki Sonuçlar (Table 3):**
- **LaneLM-512 (0-kp):** 75.07% F1
- **LaneLM-512 (2-kp):** 78.36% F1 (+3.29%)
- **LaneLM-512 (4-kp):** 81.73% F1 (+6.66%)

**Çözüm:**
- CLRNet'ten ilk 2 keypoint al
- Bu keypoint'leri model'e prompt olarak ver
- Model'in başlangıç belirsizliği azalır → zigzagging azalır

### 3. Model Smoothness Öğrenmiyor

**Sorun:**
- Training'de smoothness loss yok
- Model zigzag pattern öğreniyor
- Post-processing (smoothing, HR) yeterli değil

**PDF'de Ne Diyor? (Line 1619-1623):**
> "Analysis on hallucination. Current large language models are still struggling with hallucination. Figure 6(a) shows hallucination in LaneLM. Eq. 10 endows the model with the capability of VQA but it makes it easier for the model to predict cyclic sequences. LaneLM has learned the contextual representation of abrupt change points and consequently results in hallucination."

**Çözüm:**
- Smoothness loss (geometric, second derivative)
- Model seviyesinde smoothness zorunlu

## Kalıcı Çözüm Planı (PDF'ye Göre)

### ÇÖZÜM 1: Tokenization Granularity Artırma ⭐ EN ÖNCELİKLİ

**PDF Standard:**
- `nbins_x: 200 → 800` (PDF'de kullanılan değer)
- **Granularity: 4px → 1px per bin**
- **Beklenen Etki:** Zigzagging → Smooth predictions

**Neden Bu En Kalıcı?**
1. **Model Seviyesinde:** Post-processing değil, architecture değişikliği
2. **PDF'de Kanıtlanmış:** 800 bins kullanılıyor, başarılı sonuçlar
3. **Kök Nedeni Çözer:** Kaba quantization → zigzagging

**Trade-off:**
- ✅ Daha smooth predictions
- ✅ Küçük hatalar birikmez
- ✅ Model seviyesinde çözüm
- ❌ Model'i yeniden eğitmek gerekiyor
- ❌ Vocabulary size artıyor (200 → 800)
- ❌ Training biraz daha yavaş olabilir

### ÇÖZÜM 2: Prompting Strategy (CLRNet'ten İlk 2 Keypoint) ⭐ PDF'DE KRİTİK

**PDF'de Ne Diyor? (Line 497-499):**
> "A regression network is employed to provide the two initial keypoints, for each lane. LaneLM is responsible for completing the remaining keypoints."

**Uygulama:**
1. CLRNet'ten her lane için ilk 2 keypoint al
2. Bu keypoint'leri tokenize et
3. Model'e prompt olarak ver
4. Model kalan keypoint'leri tamamlar

**Beklenen Etki:**
- Başlangıç belirsizliği azalır
- Zigzagging azalır (PDF'de +3-6% F1 artışı)
- Model'in başlangıç noktası doğru olur

**Neden Bu Kalıcı?**
- Model seviyesinde çözüm (inference stratejisi)
- PDF'de kanıtlanmış (Table 3)
- Başlangıç belirsizliğini çözer

### ÇÖZÜM 3: Smoothness Loss (Training Strategy)

**Uygulama:**
- Geometric smoothness loss (second derivative)
- Model seviyesinde smoothness zorunlu

**Beklenen Etki:**
- Model smooth pattern öğrenir
- Zigzagging azalır

## Öncelik Sırası

1. ⭐ **nbins_x: 200 → 800** (PDF standard, en kalıcı çözüm)
2. ⭐ **Prompting Strategy** (CLRNet'ten ilk 2 keypoint, PDF'de kritik)
3. **Smoothness Loss** (Training strategy, model seviyesinde smoothness)

## Uygulama Planı

### Faz 1: Tokenization Granularity (En Öncelikli)

1. **Training Script:** `nbins_x = 800` (PDF standard)
2. **Test Config:** `nbins_x = 800` (training ile match)
3. **Model'i Yeniden Eğit:** 800 bins ile
4. **Test Et:** Sonuçları karşılaştır

### Faz 2: Prompting Strategy (PDF'de Kritik)

1. **CLRNet Head Entegrasyonu:** CLRNet'ten keypoint al
2. **İlk 2 Keypoint Tokenize Et:** Model'e prompt olarak ver
3. **Autoregressive Decode Güncelle:** Prompt ile başla
4. **Test Et:** Sonuçları karşılaştır

### Faz 3: Smoothness Loss (Training Strategy)

1. **Smoothness Loss Ekle:** Training script'e
2. **Model'i Yeniden Eğit:** Smoothness loss ile
3. **Test Et:** Sonuçları karşılaştır

## Beklenen Etki

### Önceki Durum (200 bins, no prompting)
- Granularity: 4px per bin
- Zigzagging: Yüksek
- F1@0.5: 0.02

### Sonraki Durum (800 bins, 2-kp prompting)
- Granularity: 1px per bin
- Zigzagging: Çok Düşük
- F1@0.5: 0.3-0.5 (PDF'de 81.73% F1@50)

## Not

PDF'de açıkça belirtiliyor:
- **Line 570:** "800 nbins and 100 training epochs"
- **Line 1641:** "nbins=800 aligns with our intuition"
- **Table 7:** 800 bins en iyi sonuç veriyor
- **Table 3:** 2-kp prompting +3-6% F1 artışı sağlıyor

Bu çözümler **kalıcı** çünkü:
- Model seviyesinde değişiklikler
- PDF'de kanıtlanmış
- Kök nedeni çözüyorlar








