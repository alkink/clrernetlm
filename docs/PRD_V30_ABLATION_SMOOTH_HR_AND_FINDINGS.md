## Amaç
F1@0.5’in çok düşük kalmasının kök nedenini **postprocess kaynaklı mı (smoothing / HR) yoksa model öğrenmesi kaynaklı mı** diye izole etmek.

## Yapılan Değişiklikler (Kod)
### 1) Tokenizer smoothing kontrolü
- `libs/models/lanelm/tokenizer.py`: smoothing yalnızca `smooth=True` iken uygulanacak şekilde düzeltildi (V28).
- `libs/models/detectors/lanelm_detector.py`: `decode_cfg.smooth` → `self.decode_smooth`.
- `configs/lanelm/lanelm_v4_culane_test_v26_train100.py`: `smooth=False` (eval default).
- `configs/lanelm/lanelm_v4_culane_test_v26_train100_smooth.py`: `smooth=True` ablation config’i eklendi.

### 2) HR (Hallucination Removal) toggle + sıralama
- `libs/models/detectors/lanelm_detector.py`:
  - `decode_cfg.use_hr` ve `decode_cfg.hr_min_points` eklendi.
  - HR uygulanırken sequence yönüne bağlı “bottom’un kırpılmasını” önlemek için HR **bottom→top** (y desc) sırada uygulanıp sonra tekrar y asc sıraya geri alınıyor.
  - HR içinde `np.argmax` belirsizliği giderildi: `np.where(diff>θ)` ile ilk ihlal bulunuyor.
- `configs/lanelm/lanelm_v4_culane_test_v26_train100_nohr.py`: `use_hr=False` ablation config’i eklendi.

## Deneyler (Aynı Checkpoint ile)
Kullanılan checkpoint:
- `work_dirs/v26_train100_0kp/lanelm_v4_best.pth`

Test subset:
- `dataset/list/test_100.txt`

Test runner:
- `tools/test_lanelm_runner.py` (LaneLMDetector ckpt’yi config’ten içeriden yüklüyor)

### A) smooth=False (baseline)
Sonuçlar:
- **F1@0.1 = 0.3756**
- **F1@0.5 = 0.0165**
- **F1@0.75 = 0.0000**

### B) smooth=True
Sonuçlar:
- **F1@0.1 = 0.4020** (arttı)
- **F1@0.5 = 0.0165** (değişmedi)
- **F1@0.75 = 0.0000**

Yorum:
- Smoothing görsel/gevşek eşleşmeyi artırabiliyor (IoU 0.1), ancak **strict IoU (0.5+) çöküşünü açıklamıyor**.

### C) HR kapalı (use_hr=False)
Sonuçlar:
- **F1@0.1 = 0.3460** (düştü)
- **F1@0.5 = 0.0165** (değişmedi)
- **F1@0.75 = 0.0000**

Yorum:
- HR’ın kapatılması strict IoU’yu düzeltmedi. HR burada “ana kök neden” değil.

## Tokenizer Round-Trip Sanity (Model bağımsız)
`tools/debug_training_sample_iou.py` üzerinden, `train_100` sample 0 (GT var) için:
- valid_steps=30/40
- **mean_abs_err ≈ 0.276 px**
- **max_abs_err ≈ 0.495 px**

Yorum:
- Quantize/dequantize + spline sampling hatası **çok küçük**. Tokenizer “zigzag”ın ana nedeni değil.

## Ara Sonuç (Dürüst)
Bu ablation’lar şunu gösteriyor:
- Smoothing ve HR ayarları **F1@0.5’i düzeltmiyor**.
- Tokenizer round-trip hatası ihmal edilebilir.
- Dolayısıyla F1@0.5 çöküşünün ana sebebi büyük ihtimalle **modelin genel olarak öğrenememesi / yeterli eğitim yapılmaması / training setup**.
  - Özellikle `--overfit-size 8` ile eğitilmiş bir checkpoint’in `test_100`’de yüksek F1@0.5 vermesi beklenmez.

## Sonraki Adım Önerisi
1) “Gerçek eğitim”: `train_100` üzerinde **overfit-size=100** (en az) ve PDF’e hizalı decoder ile yeni checkpoint.
2) Token accuracy (train) + train/test token match + tek-sample IoU debug ile birlikte tekrar ölçüm.


