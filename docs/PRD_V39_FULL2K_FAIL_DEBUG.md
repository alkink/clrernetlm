## V39 Full-2K (overfit-size=0) – Debug Sonuçları

### Run’lar
- HR ON: `work_dirs/v39_test100_full2k_pad01/20251225_050803`
- HR OFF: `work_dirs/v39_test100_full2k_pad01_nohr/20251225_080442`

### Metrikler
**HR ON**
- F1@0.1 = 0.3759
- F1@0.5 = 0.0723
- F1@0.75 = 0.0096

**HR OFF**
- F1@0.1 = 0.3750
- F1@0.5 = 0.0673
- F1@0.75 = 0.0096

### Boş prediction dosyaları (0 byte)
- HR ON: empty_files = 13 / 100
- HR OFF: empty_files = 13 / 100

✅ Sonuç: Bu run’da problem HR değil; HR kapatmak metrikleri düzeltmiyor ve boş output sayısını azaltmıyor.

### Eğitim tarafı sinyali (root-cause)
`work_dirs/v39_train2k_full_pad01/train.log`:
- Training loss hızlı düşüyor ve `ACC` 1.0’a yaklaşıyor
- Ancak `visualize` token diagnostik:
  - TF_ACC ≈ 0.891
  - AR_ACC ≈ 0.588
  - AR_MAE_tok ≈ 163.85

Bu, modelin autoregressive decoding’de ciddi “drift” yaşadığını (exposure bias / AR instabilite) gösterir.
Testteki “gap” ve düşük F1@0.5 ile tutarlı.

### Aksiyon Planı (V40 önerisi)
Full-2k koşusunda AR stabilizasyonu için:
- Scheduled sampling aç: `--ss-max-prob 0.2`
- AR rollout loss aç: `--ar-rollout-max-weight 0.05 --ar-rollout-min-weight 0.02`
- Padding/EOS öğretimini güçlendir: `--pad-loss-weight 1.0`
- Epoch: en az 100 (50 kısa kalıyor)


