# V13: Root Cause Analysis - F1@0.5 = 0.0000 Sorunu

## Test Sonuçları

### Test 1 (overfit-size 1):
- **F1@0.5: 0.0527** (TP: 16, FP: 384, FN: 191)
- **F1@0.1: 0.5206** (TP: 158, FP: 242, FN: 49)

### Test 2 (overfit-size 8):
- **F1@0.5: 0.0000** (TP: 0, FP: 400, FN: 207) ⚠️ **KRİTİK**
- **F1@0.1: 0.4448** (TP: 135, FP: 265, FN: 72)

## Görsel Analiz

Görsellerden tespit edilen sorunlar:
1. **Çok fazla hallucination**: Yol dışına çıkan çizgiler (kırmızı, sarı çizgiler)
2. **Yolları iyi coverlamıyor**: Şeritler gerçek yolları takip etmiyor
3. **Çok fazla false positive**: 400 FP (her sample için 4 lane slot, hepsi dolu)
4. **Geometrik bozukluk**: Çizgiler zigzag, düzgün değil

## PDF Analizi (Sayfa 879-885)

### "*" Versiyon Problemi:
> "In the * version, LaneLM underperforms CLRNet because, in Eq. 10, LaneLM actually predict pseudo-labels from CLRNet i.e. the knowledge of this part in LaneLM is distilled from the CLRNet."

### "Abrupt Change Points" Problemi:
> "LaneLM with fewer keypoint prompts is worse than the * version because, in the training sequence, a sudden jump occurs at the junction between the pseudo-label and the ground truth (see Eq. 10), which disrupts the contextual semantic information and confuses the model. It has been observed that the model often hallucinates on the side lanes, indicating that the model struggles to cope with abrupt changes in semantic information."

### Hallucination Analizi (Sayfa 1622):
> "LaneLM has learned the contextual representation of abrupt change points and consequently results in hallucination."

## Root Cause: Training/Test Mismatch

### Mevcut Training Stratejisi (V12):
1. **Lq (CLRNet pseudo-label)**: İlk 2 keypoint, noise ile
2. **Lgt (Ground Truth)**: Kalan keypoint'ler
3. **Loss Mask**: 
   - Lq için: `False` (loss hesaplanmıyor, sadece 0.1 weight ile)
   - Lgt için: `True` (loss hesaplanıyor, 1.0 weight ile)
4. **Model**: Lq'yu görüyor ama öğrenmiyor (loss çok düşük)

### Test Stratejisi:
1. **CLRNet'ten Lq alıyor**: İlk 2 keypoint
2. **Model Lq'yu yorumlayamıyor**: Çünkü training'de Lq'yu öğrenmedi
3. **"Abrupt change points" öğreniyor**: Lq → Lgt geçişindeki ani değişim
4. **Hallucination**: Model bu "abrupt change" pattern'ini yanlış yerde uyguluyor

## Çözüm Stratejileri

### 1. Lq Loss Weight'ini Artır
- **Mevcut**: 0.1 (çok düşük)
- **Önerilen**: 0.5 veya daha fazla
- **Amaç**: Model Lq'yu daha iyi öğrensin

### 2. Training Visualization Geliştir
- Lq ve Lgt'yi ayrı renklerle göster
- "Abrupt change points" görselleştir
- Model tahminlerini Lq ve Lgt ile karşılaştır

### 3. Test Visualization Ekle
- Test script'ine visualization ekle
- CLRNet Lq, model tahminleri, GT'yi göster
- Hallucination pattern'lerini görselleştir

### 4. PDF "(2-kp)" Stratejisini Dene
- GT keypoint'lerden noise ile Lq oluştur
- CLRNet yerine GT kullan (training'de)
- Test'te hala CLRNet kullan

### 5. Loss Mask Stratejisini Değiştir
- Lq için de loss hesapla (daha yüksek weight ile)
- Smooth transition loss ekle (Lq → Lgt geçişi için)

## Sonraki Adımlar

1. ✅ Root cause analizi tamamlandı
2. ⏳ Lq loss weight'ini artır (0.1 → 0.5)
3. ⏳ Training visualization geliştir (Lq/Lgt ayrı renkler)
4. ⏳ Test visualization ekle
5. ⏳ PDF "(2-kp)" stratejisini dene (opsiyonel)






