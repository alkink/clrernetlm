# V6: Presence Head Debug Analizi

## Durum

Presence head aktif edilmiş olmasına rağmen sonuçlar değişmedi:
- **F1@0.5 = 0.0428** (aynı)
- **FP = 387** (aynı)
- **Her zaman 4 lane tahmin ediliyor** (aynı)

## Olası Nedenler

### 1. Presence Head Çalışmıyor
- `presence_logits` None olabilir
- `use_presence_filter` False olabilir
- Presence head model'de yok veya yüklenmemiş olabilir

### 2. Threshold Çok Yüksek
- Tüm lane'ler threshold'u geçiyor olabilir
- Presence logits'leri çok yüksek olabilir

### 3. Presence Head Öğrenmemiş
- Model presence head'i öğrenmemiş olabilir
- Training'de presence loss çok düşük olabilir

## Yapılan Düzeltmeler

### 1. Presence Logits Debug Eklendi
- Her lane için presence_prob ve presence_logit loglanıyor
- Presence filtering sonuçları loglanıyor
- Threshold ve geçen lane sayısı loglanıyor

### 2. Presence Filtering Debug Eklendi
- Her lane için presence score ve pass/fail durumu loglanıyor
- Toplam geçen lane sayısı loglanıyor

## Beklenen Debug Çıktısı

Test script'ini çalıştırdığımızda şunları göreceğiz:
```
[DEBUG] Lane 0: presence_prob=0.XXXX, presence_logit=X.XXXX
[DEBUG] Lane 1: presence_prob=0.XXXX, presence_logit=X.XXXX
[DEBUG] Lane 2: presence_prob=0.XXXX, presence_logit=X.XXXX
[DEBUG] Lane 3: presence_prob=0.XXXX, presence_logit=X.XXXX
[DEBUG] Presence filtering: threshold=0.3
[DEBUG]   Lane 0: score=0.XXXX, pass=True/False
[DEBUG]   Lane 1: score=0.XXXX, pass=True/False
[DEBUG]   Lane 2: score=0.XXXX, pass=True/False
[DEBUG]   Lane 3: score=0.XXXX, pass=True/False
[DEBUG]   Total lanes passing filter: X/4
```

## Sonraki Adımlar

1. ✅ Presence logits debug eklendi
2. ⏳ Test script'ini çalıştır ve presence logits'lerini kontrol et
3. ⏳ Eğer presence logits'leri çok yüksekse, threshold'u düşür (0.2, 0.1)
4. ⏳ Eğer presence logits'leri çok düşükse, presence head'in öğrenip öğrenmediğini kontrol et








