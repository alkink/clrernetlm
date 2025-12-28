# V6: Empty Predictions Analizi

## Problem: TP=0, FP=0, FN=207

**Durum:** Presence filtering kapalıyken bile hiç lane tahmin edilmiyor.

## Olası Nedenler

### 1. Token Decode Sorunu
- `x_tokens_all` shape'i yanlış olabilir
- Tüm token'lar padding (0) olabilir
- Model hiç valid token üretmiyor olabilir

### 2. Filtreleme Çok Agresif
- `non_pad_mask.sum() < 2` kontrolü çok sıkı
- `decode_single_lane` hiç nokta döndürmüyor
- `coords_to_lane_normalized` None döndürüyor

### 3. Model Output Sorunu
- Model hiç token üretmiyor (tüm output'lar 0/padding)
- Autoregressive decode düzgün çalışmıyor
- Visual tokens encode edilmiyor

## Debug Adımları

### 1. Debug Script Çalıştır
```bash
python tools/debug_empty_predictions.py --sample-idx 0
```

Bu script:
- Bir test görüntüsü yükler
- Inference yapar
- Her adımı loglar
- Token'ları analiz eder
- Nerede filtrelendiğini gösterir

### 2. Kontrol Edilecekler
1. **x_tokens_all shape:** `(B, max_lanes, T)` olmalı
2. **Token değerleri:** Tüm token'lar 0 mu?
3. **Valid token sayısı:** Her lane için kaç valid token var?
4. **Decode sonucu:** `decode_single_lane` nokta döndürüyor mu?
5. **Lane oluşturma:** `coords_to_lane_normalized` başarılı mı?

## Beklenen Çıktı

Debug script şunları göstermeli:
- Token'ların gerçek değerleri
- Her lane için valid token sayısı
- Decode edilen koordinatlar
- Lane oluşturma başarısı/başarısızlığı

## Sonraki Adımlar

1. ✅ Debug script oluşturuldu (`tools/debug_empty_predictions.py`)
2. ⏳ Debug script'i çalıştır ve sonuçları analiz et
3. ⏳ Sorunun kaynağını bul (token decode, filtreleme, vs.)
4. ⏳ Çözümü uygula








