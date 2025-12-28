# V6: Presence Head Aktif Edildi

## Durum Analizi

Test sonuçları:
- **F1@0.1 = 0.6096** ✅ (İyi!)
- **F1@0.5 = 0.0428** ❌ (Çok kötü!)
- **F1@0.75 = 0.0000** ❌ (Felaket!)

### Sorunlar

1. **Çok fazla FP (False Positive):**
   - FP=387 @ 0.5
   - FP=400 @ 0.75
   - **Neden:** Her zaman 4 lane tahmin ediliyor (presence head kapalıydı)

2. **Y koordinatları aynı:**
   - Tüm sample'larda `Y[0.498,0.986]`
   - **Neden:** Model görsel bilgiyi kullanmıyor, sabit pattern öğrenmiş

3. **İlk timestep'lerde padding token seçiliyor:**
   - `t=0: pred_x=0, prob=0.84`
   - `t=1: pred_x=0, prob=0.81`
   - `t=2: pred_x=0, prob=0.30`
   - **Neden:** Model ilk timestep'lerde padding token seçiyor

## Yapılan Düzeltmeler

### 1. Presence Head Aktif Edildi
- `use_presence_filter=True` (config'den okunuyor)
- `presence_threshold=0.3` (daha düşük threshold, 0.5 çok agresifti)

### 2. Config Güncellemesi
- `decode_cfg` içine `use_presence_filter` ve `presence_threshold` eklendi
- `LaneLMDetector.__init__` içinde bu değerler okunuyor

### 3. Debug Logging
- İlk 3 timestep için logits değerleri loglanıyor
- Top-5 token probabilities gösteriliyor

## Beklenen Sonuçlar

Presence head aktif edildikten sonra:
- **FP azalmalı:** Gerçek lane sayısına göre filtreleme yapılacak
- **F1@0.5 artmalı:** Daha az false positive = daha yüksek precision
- **Lane sayısı değişmeli:** Her sample'da 4 lane yerine gerçek lane sayısı tahmin edilmeli

## Sonraki Adımlar

1. ✅ Presence head aktif edildi
2. ⏳ Test script'i çalıştır ve sonuçları karşılaştır
3. ⏳ Eğer hala FP yüksekse, threshold'u daha da düşür (0.2, 0.1)
4. ⏳ Y koordinatlarının neden aynı olduğunu araştır (visual conditioning sorunu olabilir)








