## V5 FAZ 5: İnference Optimizasyonu (Visual-First Decode + Smoothing)

### 1. Visual-First Decode
- **Dosya:** `tools/train_lanelm_v4_fixed.py`
- Yeni helper: `visual_first_decode(...)`
  - Y sabit grid (`0..T-1`)
  - Her zaman adımında model, önceki tahminleri giriş olarak alıyor
  - İlk token tamamen görsel bilgiyle belirleniyor
- **Amaç:** Training/test/visualization inference mantığını hizalamak

### 2. Visualization Güncellemesi
- `visualize()` artık parallel decode kullanmıyor
- Yeni helper ile aynı inference akışı (cross-attention → self-attention) uygulanıyor
- `tokenizer.decode_single_lane(..., smooth=True)` ile smoothing varsayılan

### 3. Visual-First Decode Tüm Pipelinelarda
- **Yeni:** `libs/models/detectors/lanelm_detector.autoregressive_decode` → visual-first AR
- **Test Scriptleri:**
  - `tools/test_lanelm_culane.py`
  - `tools/debug_test_predictions.py`
- Artık hepsi aynı helper'ı kullanıyor → training / debug / test hizalandı.

### 4. Beklenen Etki
1. **Zigzag Azalması:** Parallel decode yerine AR decode → görsel+AR uyumu
2. **Tutarlılık:** Training visualization = Test pipeline
3. **Smoothing:** Savitzky-Golay filtresi (window=15) otomatik uygulanıyor

### 5. Sonraki Adım
- Resmi MMEngine test pipeline'ında (tools/test.py + configs) aynı helper'ı kullanarak tam hizalama (şu an custom script ve `LaneLMDetector` tarafı hazır).

**Tarih:** 2024-12-30  
**Durum:** ✓ Tamamlandı (visualize + detector + test script hizalı)

