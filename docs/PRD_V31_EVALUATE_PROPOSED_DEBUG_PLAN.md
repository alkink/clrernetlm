## Amaç
Kullanıcının paylaştığı “V31 Zigzag Sistematik Debug Planı”nı doğrulamak ve yanlış/eksik beklentileri netleştirmek.

## Planın Değerlendirmesi (Dürüst)
### Doğru olanlar
- **Step 2 (token prediction error)** ve **Step 3 (train vs inference token compare)** yaklaşımı doğru: “öğrenme mi / mismatch mi” ayrımını hızlı yapar.
- **H3 (coordinate transform kayması)** hipotezi genel olarak anlamlı, ama yalnız round-trip px error ile değil, **eval hattı (Lane→string→interp→draw)** üzerinden de doğrulamak gerekir.

### Hatalı / eksik olanlar
- “Step 1’de IoU ≈ 1.0 olmalı” beklentisi **garanti değil**.
  - GT noktaları seyrek, tokenizer sabit T=40 sample yapıyor.
  - CULane metric kendi interpolation + width çizimi yapıyor.
  - GT’de görüntü dışı x’ler bulunabiliyor; tokenizer clamp/skip yapar.
  - Bu yüzden Step 1 bir “upper bound/sanity” testidir; 1.0 beklemek doğru değil.
- **H2 (nbins_x=800 yetmez)**: 800 bin, resized 800px genişlikte pratikte ~1px granülerlik demek; bu hipotez zayıf.

## Step 1 – Uygulama ve Kanıt
Yeni script eklendi:
- `tools/debug_gt_roundtrip_iou.py`

Bu script model kullanmadan:
`dataset GT(resized) → encode → decode → coords_to_lane_normalized → CULane format`
ve original GT (.lines.txt) ile `culane_metric` kıyaslaması yapar.

### test_100 üzerinde 10 GT-örneği sonucu
Not: test_100 başında “noline” örnekleri var; script GT olanları bulana kadar tarar.

`smooth=False`:
- IoU=0.1: **F1=1.0000**
- IoU=0.5: **F1=0.8667**
- IoU=0.75: **F1=0.4000**

`smooth=True`:
- IoU=0.1: **F1=1.0000**
- IoU=0.5: **F1=0.8667**
- IoU=0.75: **F1=0.3667**

Yorum:
- Pipeline (tokenizer + coord transform + metric format) genel olarak **sağlam**.
- Bu sonuçlar, “F1@0.5 düşük” probleminin kök nedeninin çoğunlukla **model prediction/eğitim** tarafında olduğunu güçlendirir.


