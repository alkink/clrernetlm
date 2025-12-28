# V13: Kritik Bulgular - "Pred: 4 lanes" vs "GT: 0 lanes" Sorunu

## Test Sonuçları

### Test (overfit-size 8):
- **F1@0.5: 0.0363** (TP: 11, FP: 388, FN: 196)
- **F1@0.1: 0.4884** (TP: 148, FP: 251, FN: 59)

## Görsel Analiz - KRİTİK BULGU

Görsellerden tespit edilen **en kritik sorun**:
- **"Pred: 4 lanes" vs "GT: 0 lanes"** - Model her zaman 4 lane predict ediyor!
- GT: 0 lanes olan durumlarda bile model 4 lane predict ediyor
- Bu yüzden FP çok yüksek (388 FP = her image için ~4 FP)

## Root Cause Analizi

### 1. Presence Filter Kapalı
```python
# configs/lanelm/lanelm_v4_culane_test.py line 52
use_presence_filter=False,  # V6: Disable until model is retrained with valid_ratio fix
```
- Presence filter kapalı olduğu için model her zaman 4 lane slot'u dolduruyor
- Presence head çalışmıyor, tüm lane'ler predict ediliyor

### 2. Training'de GT: 0 Lanes Durumları Yok
- Training'de sadece lane slot'ları için negative lanes var (presence_target=0.0)
- Ama **GT: 0 lanes olan image'ler training'e dahil değil**
- Model "no lane" durumunu öğrenmemiş

### 3. Model Her Zaman max_lanes Kadar Predict Ediyor
- `autoregressive_decode` her zaman max_lanes (4) kadar lane decode ediyor
- Presence filter kapalı olduğu için hiçbir lane filtrelenmiyor
- Sonuç: Her image için 4 lane predict ediliyor

## Çözüm Stratejileri

### 1. Presence Filter'ı Aç ve Threshold Optimize Et
- `use_presence_filter=True` yap
- Threshold'u optimize et (0.3-0.7 arası deneyebiliriz)
- Presence head'in çalıştığından emin ol

### 2. Training'e GT: 0 Lanes Durumlarını Ekle
- GT: 0 lanes olan image'leri training'e ekle
- Bu image'ler için tüm lane slot'ları negative olacak (presence_target=0.0)
- Model "no lane" durumunu öğrensin

### 3. Presence Loss'u Güçlendir
- Presence loss weight'ini artır (0.3 → 0.5 veya daha fazla)
- Negative samples için presence loss'u daha agresif yap

### 4. Test'te Presence Filter Kullan
- Config'de `use_presence_filter=True` yap
- Threshold'u optimize et

## Sonraki Adımlar

1. ✅ Root cause analizi tamamlandı
2. ⏳ Presence filter'ı aç ve threshold optimize et
3. ⏳ Training'e GT: 0 lanes durumlarını ekle
4. ⏳ Presence loss'u güçlendir
5. ⏳ Modeli yeniden eğit
6. ⏳ Test et ve sonuçları analiz et






