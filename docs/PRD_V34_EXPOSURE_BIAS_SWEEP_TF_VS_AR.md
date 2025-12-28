## Amaç
“ACC=1 ama zigzag var → exposure bias” hipotezini tek örnek yerine çoklu örnek üzerinde test edebilmek.

## Değişiklik
Dosya: `tools/visualize_tf_vs_ar.py`
- Artık tek örnek yerine sweep destekliyor:
  - `--num-samples`: GT içeren kaç örnek işlenecek
  - `--sample-idx`: başlangıç index
  - `--save-max`: kaç örnek için TF/AR görseli kaydedilecek
- Her örnek için konsola şunları basar:
  - `TF_ACC`, `TF_MAE_tok`
  - `AR_ACC`, `AR_MAE_tok`
- Özet rapor:
  - `TF==AR` kaç örnekte?
  - `TF perfect but AR not` kaç örnekte? (exposure bias için en güçlü sinyal)

## Kullanım Önerisi
Örnek:
```bash
/home/alki/miniconda3/envs/clrernet/bin/python tools/visualize_tf_vs_ar.py \
  --lanelm-ckpt work_dirs/v26_train100_0kp/lanelm_v4_best.pth \
  --list-path dataset/list/train_100.txt \
  --sample-idx 0 --num-samples 20 --save-max 10 --device cuda --save-dir work_dirs/_tf_ar_sweep
```


