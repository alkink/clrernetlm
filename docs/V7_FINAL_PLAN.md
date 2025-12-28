# V7: Zigzagging Kalıcı Çözüm - Final Plan

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

## Zigzagging'in Kök Nedeni: Kaba Tokenization

### Mevcut Durum
- **nbins_x = 200**
- **Granularity: 800px / 200 = 4px per bin**
- **Sorun:** Her token 4px'lik bir aralığı temsil ediyor → zigzagging

### PDF'de Ne Kullanılıyor?
- **nbins_x = 800** (Section 4.1)
- **Granularity: 800px / 800 = 1px per bin**
- Çok daha ince granularity → smooth predictions

## Kalıcı Çözüm: Tokenization Granularity Artırma

### Uygulama

1. **Training Script:**
   - `nbins_x: 200 → 400` (2px per bin)
   - Veya `200 → 800` (1px per bin, PDF standard)

2. **Test Config:**
   - `nbins_x: 200 → 400` (training ile match)

3. **Model Architecture:**
   - Otomatik güncelleniyor (nbins_x parametresi)

### Neden Bu En Kalıcı Çözüm?

1. **Model Seviyesinde:**
   - Post-processing değil, model architecture değişikliği
   - Model smoothness öğrenebilir
   - Küçük hatalar birikmez

2. **PDF'de Kanıtlanmış:**
   - PDF'de 800 bins kullanılıyor
   - Başarılı sonuçlar

3. **Kök Nedeni Çözer:**
   - Kaba quantization → zigzagging
   - İnce granularity → smooth predictions

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

## Yapılan Değişiklikler

1. ✅ **Training Script:** `nbins_x = 400` (train_lanelm_v4_fixed.py)
2. ✅ **Test Config:** `nbins_x = 400` (lanelm_v4_culane_test.py)
3. ✅ **HR Algoritması:** Implement edildi (post-processing)
4. ⏳ **Model Eğitimi:** 400 bins ile yeniden eğitilmeli

## Sonraki Adımlar

1. ✅ Training script güncellendi
2. ✅ Test config güncellendi
3. ⏳ **Model'i yeniden eğit** (400 bins ile) ⭐ KRİTİK
4. ⏳ Test et ve sonuçları karşılaştır
5. ⏳ Gerekirse 800 bins'e çıkar (PDF standard)

## Not

Bu değişiklik mevcut checkpoint ile uyumlu değil. Model'i yeniden eğitmek gerekiyor. Ancak bu **en kalıcı çözüm** çünkü:
- Model seviyesinde smoothness sağlar
- PDF'de kanıtlanmış
- Kök nedeni çözer (kaba quantization)

HR algoritması post-processing çözümü olduğu için zigzagging'in kök nedenini çözmez. **Tokenization granularity artırma** en kalıcı çözümdür.








