# V7: Zigzagging Kök Neden Analizi ve Kalıcı Çözüm

## Test Sonuçları Analizi

### HR Sonrası Durum
- **F1@0.5 = 0.0494** (HR öncesi: 0.0264, minimal iyileşme)
- **FP@0.5 = 385** (HR öncesi: 392, minimal azalma)
- **Zigzagging:** Hala devam ediyor (görsellerde sarı çizgiler)

### HR Neden Yeterli Değil?

HR algoritması **post-processing** çözümü:
- Anormal sıçramaları filtreler ✅
- Ama zigzagging'in kök nedenini çözmez ❌
- Model hala zigzag pattern üretiyor ❌

## Zigzagging'in Kök Nedenleri

### 1. Tokenization Granularity Çok Kaba ⭐ EN KRİTİK

**Mevcut:**
- `nbins_x = 200`
- `img_w = 800`
- **Granularity: 800 / 200 = 4px per bin**

**Sorun:**
- Her token 4px'lik bir aralığı temsil ediyor
- Küçük hatalar (1-2px) birikiyor
- Model smoothness öğrenemiyor (çok kaba quantization)

**PDF'de:**
- `nbins_x = 800` (bizde 200)
- **Granularity: 800 / 800 = 1px per bin**
- Çok daha ince granularity → daha smooth predictions

### 2. Autoregressive Decode'da Exposure Bias

**Sorun:**
- Training'de GT görüyor, inference'da kendi tahminlerini görüyor
- Küçük hatalar birikiyor → zigzagging

**Çözüm:**
- Prompting strategy (CLRNet'ten ilk 2 keypoint)
- Ama bu da geçici çözüm (model seviyesinde değil)

### 3. Model Smoothness Öğrenmiyor

**Sorun:**
- Training'de smoothness loss yok
- Model zigzag pattern öğreniyor
- Post-processing (smoothing, HR) yeterli değil

**Çözüm:**
- Smoothness loss (geometric, second derivative)
- Model seviyesinde smoothness zorunlu

## Kalıcı Çözüm: Tokenization Granularity Artırma

### Neden Bu En Kalıcı Çözüm?

1. **Model Seviyesinde:** Post-processing değil, model architecture değişikliği
2. **PDF'de Kanıtlanmış:** PDF'de 800 bins kullanılıyor
3. **Kök Nedeni Çözer:** Kaba quantization → zigzagging
4. **Smoothness Sağlar:** İnce granularity → smooth predictions

### Uygulama

**Değişiklik:**
- `nbins_x: 200 → 400` (veya 800, PDF'deki gibi)
- Model'i yeniden eğitmek gerekiyor

**Beklenen Etki:**
- Granularity: 4px → 2px (veya 1px)
- Zigzagging: Yüksek → Düşük
- F1@0.5: 0.05 → 0.3-0.5

### Trade-off

**Avantajlar:**
- Daha smooth predictions
- Küçük hatalar birikmez
- Model seviyesinde çözüm

**Dezavantajlar:**
- Model'i yeniden eğitmek gerekiyor
- Vocabulary size artıyor (200 → 400)
- Training biraz daha yavaş olabilir

## Sonraki Adımlar

1. ✅ HR algoritması implement edildi (post-processing)
2. ⏳ **Tokenization granularity artır** (model seviyesinde) ⭐ ÖNCELİK
3. ⏳ Smoothness loss ekle (training strategy)
4. ⏳ Prompting strategy (inference optimization)

## Not

HR algoritması post-processing çözümü olduğu için zigzagging'in kök nedenini çözmez. **Tokenization granularity artırma** en kalıcı çözümdür çünkü:
- Model seviyesinde smoothness sağlar
- PDF'de kanıtlanmış (800 bins)
- Kök nedeni çözer (kaba quantization)








