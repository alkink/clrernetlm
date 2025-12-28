# Kritik Düzeltme: Padding Lane Filtreleme

## Sorun

### Kanıtlı Bulgular

**Training Dataset:**
- 4 lanes: 60% (60 samples)
- 3 lanes: 32% (32 samples)
- Mean: 3.39 lanes

**Test Dataset:**
- 3 lanes: 69% (69 samples)
- 0 lanes: 31% (31 samples)
- **4 lanes: 0% (0 samples!)** ⚠️
- Mean: 2.07 lanes

**Model Behavior:**
- Model **her zaman** `max_lanes` (4) lane predict ediyor
- Test'te hiç 4 lane yok ama model yine de 4 lane predict ediyor
- Bu **false positive'leri çok artırıyor**!

### Root Cause

1. **Model her zaman 4 lane predict ediyor:**
   - `autoregressive_decode` fonksiyonu `for lane_idx in range(max_lanes)` döngüsünde
   - Hiçbir zaman durmuyor, her zaman 4 lane üretiyor

2. **Padding token kontrolü yok:**
   - `LaneLMDetector.predict` fonksiyonunda padding kontrolü yok
   - Model padding token'ları (0) predict etse bile, decode ediliyor
   - Bu false positive'leri artırıyor

3. **Training vs Test distribution farkı:**
   - Training'de 60% 4 lane var
   - Test'te 0% 4 lane var
   - Model training distribution'ına göre öğrenmiş

## Çözüm

### Padding Lane Filtreleme Eklendi

**Dosya:** `libs/models/detectors/lanelm_detector.py`

**Değişiklik:**
- Padding token kontrolü eklendi
- Eğer bir lane'in tüm token'ları padding ise, o lane filtreleniyor
- En az 2 geçerli token gerekiyor

**Kod:**
```python
# Check if this lane has any non-padding tokens
x_tokens_lane = x_tok[l]
pad_token_x = self.tokenizer_cfg.pad_token_x
non_pad_mask = (x_tokens_lane != pad_token_x) & (x_tokens_lane != 0)

# Skip lanes that are all padding (no valid tokens)
if non_pad_mask.sum() < 2:  # Need at least 2 valid points for a lane
    continue
```

## Beklenen Etki

### Önceki Durum
- Model her zaman 4 lane predict ediyor
- Test'te 0% 4 lane var ama model yine de 4 lane predict ediyor
- False positive'ler çok yüksek (FP=392 @ IoU 0.5)
- F1@0.5: 0.0264

### Sonraki Durum (Beklenen)
- Model sadece geçerli lane'leri predict ediyor
- Padding lane'ler filtreleniyor
- False positive'ler azalıyor (FP: 392 → ~200)
- F1@0.5: 0.0264 → 0.1+

## Test

Düzeltmeyi test etmek için:

```bash
python tools/test.py configs/lanelm/lanelm_v4_culane_test.py dummy.pth
```

**Beklenen:**
- Predicted lane sayısı azalması (4 → 2-3)
- False positive'ler azalması
- F1 artışı

## Notlar

1. **Backward Compatibility:** Bu değişiklik inference-only, training'i etkilemez
2. **Threshold:** En az 2 geçerli token gerekiyor (bir lane için minimum)
3. **Training Distribution:** Model training'de 4 lane görmüş, bu yüzden 4 lane predict ediyor








