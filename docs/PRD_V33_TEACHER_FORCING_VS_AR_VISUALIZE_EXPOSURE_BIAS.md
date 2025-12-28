## Amaç
“Training’de ACC=1.0 ama görselde zigzag var” iddiasını kanıtla ayırmak:
- (A) Gerçekten **exposure bias / error accumulation** mı?
- (B) Görselleştirme/postprocess hattı mı yanıltıyor?

## Gerçek Durum (Kod)
`tools/train_lanelm_v4_fixed.py` içindeki `visualize()` daha önce **teacher forcing değil**, AR decode (`visual_first_decode`) kullanıyordu.
Bu yüzden training loglarında ACC yüksek olsa bile, görselleştirme “inference-like” hataları gösterebilir.

## Yapılan Değişiklik
Dosya: `tools/train_lanelm_v4_fixed.py`
- `visualize()` içine iki ayrı decode eklendi:
  - **AR decode** (kırmızı): `visual_first_decode(...)` (inference benzeri)
  - **Teacher forcing argmax** (mavi): GT token’lardan `x_in_tf` oluşturulup tek forward ile argmax
- GT (yeşil) ile overlay edilerek iki ayrı görsel kaydediliyor:
  - `epXXXX_ar.jpg`
  - `epXXXX_tf.jpg`
- Konsola token-level diagnostik basılıyor:
  - `TF_ACC`, `TF_MAE_tok`
  - `AR_ACC`, `AR_MAE_tok`

## Beklenen Yorum
- Eğer **TF_ACC≈1.0** ve **AR_ACC düşük** ise: bu güçlü şekilde **exposure bias / rollout** problemine işaret eder.
  - Çözüm: scheduled sampling’i artırma + rollout loss’u güçlendirme (mevcut scriptte knob’lar var).
- Eğer **TF_ACC de düşük** ise: model zaten token’ları öğrenmiyor (loss/öğrenme problemi).


