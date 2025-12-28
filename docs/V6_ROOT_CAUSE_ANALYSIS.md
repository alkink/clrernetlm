# V6: Root Cause Analizi - Model Hiç Token Üretmiyor

## Kritik Bulgu

Debug logları çok açık:
```
[DEBUG] Sample 0: x_tokens_all shape: torch.Size([100, 1, 40])
[DEBUG] Sample 0: x_tok min/max: [0, 0]
[DEBUG] Lane 0: valid tokens = 0/40, x_range=[0, 0]
```

**Sorun:** Model hiç token üretmiyor - tüm output'lar 0 (padding).

## Olası Nedenler

### 1. Model Checkpoint Sorunu
- Checkpoint yanlış yükleniyor olabilir
- Model weights'leri yüklenmemiş olabilir
- Training checkpoint'i ile test checkpoint'i farklı olabilir

### 2. Model Inference Modu
- Model `eval()` modunda değil
- BatchNorm/Dropout aktif kalıyor
- Model davranışı değişiyor

### 3. Visual Tokens Encode Sorunu
- Visual tokens encode edilmiyor
- Visual tokens shape'i yanlış
- Visual tokens tüm 0 olabilir

### 4. Batch Size Sorunu
- Batch size 100 çok büyük
- Model batch dimension'ında sorun yaşıyor
- Her sample için aynı output üretiyor

### 5. Training vs Test Farkı
- Training'de teacher forcing kullanılıyor
- Test'te autoregressive decode kullanılıyor
- Model autoregressive decode'u öğrenmemiş olabilir

## Yapılan Düzeltmeler

### 1. Early Stopping Kaldırıldı
- Training'deki `visual_first_decode` ile tam uyumlu hale getirildi
- Early stopping kaldırıldı (tüm T timestep decode ediliyor)

### 2. Shape Düzeltmesi
- `all_x` boşsa `max_lanes` kadar padding lane oluşturuluyor (önceden sadece 1)

### 3. Logits Debug Eklendi
- İlk 3 timestep için logits değerleri loglanıyor
- Top-5 token probabilities gösteriliyor

### 4. Model Eval Mode
- `self.lanelm.eval()` eklendi (inference modu)

## Sonraki Adımlar

1. ✅ Early stopping kaldırıldı
2. ✅ Shape düzeltmesi yapıldı
3. ✅ Logits debug eklendi
4. ⏳ Test script'i çalıştır ve logits değerlerini kontrol et
5. ⏳ Model checkpoint'ini kontrol et (weights yükleniyor mu?)
6. ⏳ Visual tokens'ı kontrol et (encode ediliyor mu?)

## Beklenen Debug Çıktısı

Test script'i çalıştırdığında şunları görmeliyiz:
```
[DEBUG] Lane 0, t=0: pred_x=?, logits_range=[?, ?], mean=?, top5_tokens=?, top5_probs=?
[DEBUG] Lane 0, t=1: pred_x=?, logits_range=[?, ?], mean=?, top5_tokens=?, top5_probs=?
[DEBUG] Lane 0, t=2: pred_x=?, logits_range=[?, ?], mean=?, top5_tokens=?, top5_probs=?
```

Eğer:
- `pred_x=0` ve `top5_probs` tüm aynı → Model hiç öğrenmemiş
- `pred_x=0` ama `top5_probs` farklı → Model öğrenmiş ama padding token'ı seçiyor
- `pred_x>0` → Model çalışıyor, başka bir sorun var








