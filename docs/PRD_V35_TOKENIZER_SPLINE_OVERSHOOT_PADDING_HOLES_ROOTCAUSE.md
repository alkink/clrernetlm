## Amaç
“TF=AR=GT token (ACC=1.0) ama görselde devasa zigzag/çapraz çizgiler var” çelişkisinin kök nedenini tespit etmek ve düzeltmek.

## Kök Neden (Kanıtla)
`LaneTokenizer.encode_single_lane()` GT lane’leri token’a çevirirken `_fit_spline()` ile **cubic spline benzeri** bir interpolasyon kullanıyordu.

Bazı GT lane’lerde (özellikle kenar lane’lerde):
- anotasyon noktalarında **x < 0** veya **x > img_w** görülebiliyor,
- cubic interpolasyon **aşırı overshoot** yapıp birçok sample_y’de x’i görüntü dışına taşıyabiliyor,
- encode logic’i bu timestepleri **padding (x=0, y=T)** bırakıyor.

Sonuç:
- GT token hedefleri **delikli (çok sayıda pad)** hale geliyor.
- Decode/çizim sırasında bu “deliklerden” dolayı ardışık valid noktalar arasında **çok büyük y boşlukları** oluşuyor ve çizgi bu boşlukları düz segmentle birleştirince **devasa çapraz/zigzag** görünüyor.
- Bu durum, model token’ları mükemmel öğrense bile (TF_ACC=1) **görselin bozuk görünmesine** neden olur.

### Sayısal kanıt (train_100 sample0)
Eski durumda sample0’da bazı lane’lerde spline_x min/max ciddi biçimde out-of-bounds’tu (örn. yüzlerce px taşma) ve encode valid_steps düşüktü.

## Uygulanan Fix
Dosya: `libs/models/lanelm/tokenizer.py`

1) `_fit_spline()` overshoot riskini azaltacak şekilde değiştirildi:
- Tercihen **PCHIP (shape-preserving)**,
- fallback: **linear**.

2) Fitting öncesi x ve y clamp:
- `y` zaten `[0, img_h-1]`
- `x` artık `[0, img_w-1]` clamp ediliyor (out-of-bound anotasyonların spline’ı bozmasını azaltmak için).

3) Encode’da vertical support uygulanıyor:
- `y_sample` noktası GT’nin `[y_min, y_max]` aralığı dışındaysa padding kalıyor.

## Etki ve Notlar
- Bu değişiklik **training target’larını** değiştirir → **yeniden eğitim gerekir**.
- Model-free GT round-trip sanity (`tools/debug_gt_roundtrip_iou.py`) bu değişiklikten sonra test_100 üzerinde seçilen 10 GT örneğinde:
  - IoU=0.5 F1 **1.0000** (önce 0.8667 idi)
  - IoU=0.75 F1 **1.0000**
  Bu, tokenizer+coord+metric pipeline’ın artık daha “düzgün/kararlı” hedef ürettiğini gösterir.

## Sonraki Adım
- Bu tokenizer fix’iyle `train_lanelm_v4_fixed.py` üzerinden yeniden eğitim (train_100 tamamı).
- Ardından:
  - `eval_token_acc.py` ile TF ACC,
  - `test_lanelm_runner.py` ile test_100 F1@0.5
  tekrar ölçülmeli.


