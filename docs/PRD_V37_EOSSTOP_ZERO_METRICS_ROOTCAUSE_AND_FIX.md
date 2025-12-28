## Problem
`work_dirs/v37_test100_eosstop/20251224_225256` test koşusunda tüm metrikler 0 görünüyor.

## Kök Neden (Kanıt)
Log’da:
- `TP=0, FP=0, FN=207` ve JSON’da `TP/FP/FN = null`, `F1=0`.

Bu ancak şu durumda olur:
- prediction dosyaları **boş** yazılmıştır (hiç lane yok).

Doğrulama:
- `work_dirs/v26_test100/predictions/...*.lines.txt` dosyalarının bir kısmı **0 byte**.
  - Bu koşu aynı `result_dir`’ı kullandığı için eski prediction klasörünü de “boş dosyalarla” overwrite etmiş.

## Neden lane’ler tamamen kayboldu?
EOS-stop (x=0) mekanizması “çok erken” devreye girerse:
- lane decode 1–2 adımdan sonra durur
- `valid_count < 2` filtresi yüzünden lane tamamen düşer
- sonuçta tüm örneklerde boş output oluşur → TP/FP=0

## Fix
### 1) EOS-stop için guard eklendi
- `eos_min_t`: t<5 iken EOS sayma
- `eos_min_valid`: en az 2 non-zero token görülmeden EOS stop’a izin verme

Dosyalar:
- `libs/models/lanelm/tokenizer.py`: yeni config alanları
- `libs/models/detectors/lanelm_detector.py`: EOS-stop mantığı + decode_cfg→tokenizer_cfg propagation
- `configs/lanelm/lanelm_v4_culane_test_v26_train100.py`: yeni parametreler

### 2) Güvenli test config’i
Prediction overwrite’ı önlemek için ayrı result_dir kullanan yeni config:
- `configs/lanelm/lanelm_v4_culane_test_v37_eosstop_safe.py`

## Sonraki Adım
Bu yeni config ile test_100 tekrar koşturulmalı; hedef:
- prediction dosyaları boş değil
- F1@0.1/0.5 eski baseline’dan tamamen 0’a düşmüyor


