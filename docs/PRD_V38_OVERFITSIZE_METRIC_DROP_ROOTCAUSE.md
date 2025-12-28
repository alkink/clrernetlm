## Durum
Senin paylaştığın 3 test koşusu:
- `v37_test100_eosstop_safe/20251225_021627`
- `v37_test100_eosstop_safe/20251225_022103`
- `v37_test100_eosstop_safe/20251225_023111`

EOS-stop “safe” guard’ları sonrası metrikler artık **0’a düşmüyor** ve görsel olarak da çok daha stabil.

## Bu 3 run’ın metrik özeti
JSON/loglardan:

- **021627**
  - F1@0.1 = **0.6227**
  - F1@0.5 = **0.1153**
  - TP0.5=35 FP0.5=365 FN0.5=172

- **022103**
  - F1@0.1 = **0.6194**
  - F1@0.5 = **0.1087**
  - TP0.5=33 FP0.5=367 FN0.5=174

- **023111**
  - F1@0.1 = **0.6293**
  - F1@0.5 = **0.0231**
  - TP0.5=7 FP0.5=393 FN0.5=200

Önemli gözlem: **F1@0.1 benzer**, ama **F1@0.5** üçüncü koşuda ciddi düşüyor → “yakın ama yeterince hassas değil” (strict IoU kaçıyor).

## “Overfit-size arttıkça düşüyor” niye oluyor?
Bu üç testin `vis_data/config.py`’sinde **ckpt_path aynı** görünüyor:
- `work_dirs/v36_train100_tokenfix_padloss/lanelm_v4_best.pth`

Ama bu path **testler arasında overwrite edilmiş**:
- `lanelm_v4_best.pth` mtime: **2025-12-25 02:26:57**
- `023111` testi **02:32** civarı çalışmış → bu test **yeni overwrite edilen checkpoint’i** kullanıyor.

Yani pratikte:
- 021627 ve 022103 → **eski checkpoint**
- 023111 → **yeni checkpoint (02:26 sonrası)**

Bu yüzden düşüş “overfit-size”tan ziyade, **farklı eğitilmiş checkpoint’lerin** karşılaştırılması.

## Yeni checkpoint neden daha kötü (F1@0.5)?
`work_dirs/v36_train100_tokenfix_padloss/train.log`:
- Dataset size: **8**
- “X-LOSS ONLY” (Y-loss disabled)
- 100 epoch sonunda:
  - **Best loss ~0.1417** (overfit için hâlâ yüksek)
  - TF_ACC ~0.967 ama TF_MAE_tok ~12.9
  - AR_ACC ~0.04–0.05 (exposure bias ciddi)

Bu “8-image overfit” koşusu **gerçek overfit değil** → model 8 görüntünün tamamını aynı anda çok iyi fit edememiş (underfit/kompromi).
Bu da testte strict IoU (0.5) için hassasiyet kaybı olarak yansıyor.

## Önerilen düzeltme (kıyaslamayı doğru yapmak için)
1) Her overfit-size için ayrı `--work-dir` kullan (overwrite yok):
   - `work_dirs/v38_overfit_1/...`
   - `work_dirs/v38_overfit_4/...`
   - `work_dirs/v38_overfit_8/...`
2) Test config’te `ckpt_path`’i açıkça o run’ın checkpoint’ine bağla.
3) Overfit-size büyüdükçe aynı kaliteyi bekliyorsak:
   - epoch artır (veya LR schedule’u değiştir)
   - pad/EOS öğretmek için `pad_loss_weight > 0` kullan
   - exposure bias için scheduled sampling / AR rollout weight’i kontrollü aç


