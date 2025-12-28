## Amaç
Görsellerde görülen “devasa loop/çapraz/zigzag” artefaktlarını azaltmak ve F1@0.5’i iyileştirmek için:
- padding bölgelerinde modelin rastgele x üretmesini engellemek
- inference’ta EOS/pad geldiğinde decode’u durdurmak

## Bulgular
TF/AR görsellerde bazı örneklerde model tüm T adımında x üretiyor. Eğer padding bölgeleri loss ile öğretilmediyse (`pad_loss_weight=0`) model:
- GT’nin olmadığı bölgelerde rastgele x üretebilir,
- bu da çizimde ve metric string üretiminde “pollution” yaratır (özellikle alt bölgelerde).

## Uygulanan Değişiklik
### 1) EOS stop (x==0) ile erken durdurma (opsiyonel)
Dosya: `libs/models/detectors/lanelm_detector.py`
- `autoregressive_decode` içine `enable_eos_stop` ve `eos_consecutive` kontrolü eklendi.
- Varsayılanlar konservatif: 2 ardışık `x=0` gelince lane için decode durur.

### 2) Config üzerinden açma
Dosya: `libs/models/lanelm/tokenizer.py`
- `LaneTokenizerConfig` içine:
  - `enable_eos_stop: bool`
  - `eos_consecutive: int`

Dosya: `configs/lanelm/lanelm_v4_culane_test_v26_train100.py`
- `decode_cfg.enable_eos_stop=True`, `decode_cfg.eos_consecutive=2`

## Sonraki Adım (Eğitim)
EOS stop tek başına yeterli olmayabilir; çünkü modelin `x=0` üretmeyi öğrenmesi gerekir.
Bu yüzden yeni eğitim koşusunda:
- `--pad-loss-weight` > 0 (örn. 1.0) verilerek padding bölgelerinde `x=0` hedefi öğretilmeli.


