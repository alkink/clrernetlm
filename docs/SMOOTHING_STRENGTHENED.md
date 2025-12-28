# Smoothing Güçlendirme - Uygulandı

## Değişiklik

**Dosya:** `libs/models/lanelm/tokenizer.py`

**Değişiklik:**
- `decode_single_lane` fonksiyonundaki smoothing window_length artırıldı
- **Önceki:** `window_length = min(15, len(x_eval))`
- **Yeni:** `window_length = min(25, len(x_eval))`

## Neden?

### Sorun
1. Training visualization: `decode_single_lane(..., smooth=True)` → direkt çizim → smooth görünüyor ✅
2. Test prediction: `decode_single_lane(..., smooth=True)` → `coords_to_lane_normalized` → `Lane` spline → `get_prediction_string` → spline interpolation → zigzag artıyor ❌

### Root Cause
- Test prediction dosyasındaki koordinatlar zaten zigzag (std=4.9981)
- Spline interpolation zigzag'ı daha da artırıyor
- Smoothing yetersiz (window_length=15)

### Çözüm
- Smoothing window_length artırıldı (15 → 25)
- Bu, spline interpolation öncesi zigzag'ı daha iyi azaltır
- Özellikle test predictions için kritik

## Beklenen Etki

### Önceki Durum
- Test prediction zigzag: std(diffs) = 4.9981
- IoU düşük: 0.03-0.25
- F1 0.5'te: 0.0000

### Sonraki Durum (Beklenen)
- Test prediction smooth: std(diffs) < 2.0
- IoU artışı: 0.3-0.5+
- F1 0.5'te: 0.3+

## Test

Düzeltmeyi test etmek için:

```bash
python tools/test.py configs/lanelm/lanelm_v4_culane_test.py dummy.pth
```

**Beklenen:**
- Test prediction dosyalarında zigzag azalması
- IoU 0.5'te F1 artışı (0.0000 → 0.3+)
- TP artışı, FP/FN azalması

## Notlar

1. **Backward Compatibility:** Bu değişiklik tüm `decode_single_lane` çağrılarını etkiler (training, test, visualization)
2. **Performance:** Smoothing biraz daha yavaş olabilir (window_length artışı)
3. **Trade-off:** Daha güçlü smoothing → daha smooth ama belki biraz daha yumuşak (edge cases'de)








