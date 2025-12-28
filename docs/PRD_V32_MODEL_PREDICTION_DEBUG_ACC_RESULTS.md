## Amaç
“Problem model prediction’da mı?” sorusunu somut ölçmek için teacher forcing altında X-token accuracy (ACC) ve token error ölçümü.

## Araç
Yeni script:
- `tools/eval_token_acc.py`

Ölçtüğü metrikler:
- `X_token_ACC`: valid (non-pad) pozisyonlarda argmax(pred) == GT oranı
- `mean_abs_token_error`: valid pozisyonlarda \(|pred_x - gt_x|\) ortalaması (token biriminde; nbins_x=800 için ~1 token ≈ 1px)

## Checkpoint
- `work_dirs/v26_train100_0kp/lanelm_v4_best.pth`
  - cfg: T=40, nbins_x=800, embed_dim=512, num_layers=3, Full FPN (3 level)

## Sonuçlar
### train_100 (50 örnek)
- `X_token_ACC = 0.7489`
- `mean_abs_token_error = 23.909`

Yorum:
- Model **train_100 üzerinde bile** teacher forcing altında tam oturmuyor (ACC 0.75 civarı).
- Ortalama hata ~24 token ≈ **~24px** mertebesinde; bu strict IoU=0.5’i kolayca düşürür.

### test_100 (GT olan 50 örnek)
- `X_token_ACC = 0.0306`
- `mean_abs_token_error = 72.252`

Yorum:
- Genelleme çok zayıf; test setinde token tahmini neredeyse rastgele seviyesinde.

## Sonuç (Dürüst)
Bu ölçümler, “pipeline sağlam ama model F1@0.5 düşük” bulgusuyla tutarlı:
- Kök sorun **modelin yeterince öğrenmemesi / eğitim yetersizliği / genelleme problemi**.
- Smoothing/HR/tokenizer round-trip gibi postprocess bileşenleri ana sebep değil.

## Önerilen Sonraki Adım
1) `overfit-size=100` ile gerçek eğitim (train_100 tamamı) + epoch sayısı yeterli.
2) Eğitim boyunca `ACC` hedefi: train tarafında **>0.95**.
3) Ardından test_100’de tekrar ölçüm (F1@0.5 toparlıyor mu?).


