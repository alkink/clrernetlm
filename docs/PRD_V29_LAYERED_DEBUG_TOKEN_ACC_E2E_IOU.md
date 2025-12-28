## Amaç
LaneLM’in “puanlar çok düşük / F1@0.5 çökük” problemini **katman katman izole ederek** kök nedeni bulmak.

## Önerilen Katmanlı Strateji
Training → (1) Token Accuracy → (2) Train vs Test Token Match → (3) Coord Transform → (4) CULane Metric

Bu stratejinin mantığı:
- (1) düşükse: model/loss/öğrenme problemi
- (1) yüksek ama (2) düşükse: inference mismatch / decoding farkı
- (1)(2) yüksek ama (3)(4) düşükse: coordinate transform + eval pipeline sorunu (en sık görülen)

## Mevcut Durum: Token Accuracy Zaten Loglanıyor
`tools/train_lanelm_v4_fixed.py` çıktısında `ACC=...` var.
Not: stdout’a basıldığı için grep ile yakalamak adına çalıştırmayı `tee` ile log dosyasına yazmak önerilir.

## E2E IoU Debug Geliştirmesi
Dosya: `tools/debug_training_sample_iou.py`
- `--smooth` bayrağı eklendi: tokenizer decode smoothing’i aç/kapat ablation.
- “round-trip sanity” eklendi: GT (resized) → tokenize → dequantize → spline(sample_ys) ile kıyas
  - Bu metrik model bağımsızdır ve yalnız tokenizer quantization error’ını ölçer.

## Smoothing Ablation için Test Config’i
Dosya: `configs/lanelm/lanelm_v4_culane_test_v26_train100_smooth.py`
- `decode_cfg.smooth=True`
- çıktı klasörü ayrıldı: `work_dirs/v26_test100_smooth/predictions`

## Beklenen Sonuç
- Eğer `smooth=False` ile F1@0.5 belirgin artarsa: problem büyük oranda postprocess geometri değişimi kaynaklıdır.
- Artmazsa: bir sonraki adım coord transform + eval pipeline’ı (crop/scale/normalize) adım adım ölçmek.


