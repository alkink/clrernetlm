## Amaç
CULane metriklerinde özellikle **F1@0.5’in çökmesi** gibi “strict IoU” hassasiyetlerine sebep olabilecek **inference/postprocess smoothing** davranışını kontrol altına almak ve ablation yapmayı mümkün kılmak.

## Bulgular (Dürüst Kök Neden)
`libs/models/lanelm/tokenizer.py` içinde `decode_single_lane(..., smooth: bool)` imzası olmasına rağmen smoothing bloğu **`smooth` parametresini dikkate almıyordu** ve Savitzky–Golay filtresi **her zaman** uygulanıyordu.

Bu iki yüzden kritik:
- **Model doğru token’ı üretse bile**, de-quantize sonrası smoothing geometriyi değiştirip GT ile IoU’yu düşürebilir.
- Metrikler ve görseller “model kötü” gibi görünebilir; aslında postprocess hatası olabilir.

## Uygulanan Değişiklikler
### 1) Tokenizer smoothing kontrolü düzeltildi
Dosya: `libs/models/lanelm/tokenizer.py`
- SavGol smoothing artık **yalnızca** `smooth=True` iken çalışıyor.

### 2) Inference’te smoothing config ile kontrol ediliyor
Dosya: `libs/models/detectors/lanelm_detector.py`
- `decode_cfg.smooth` okunuyor → `self.decode_smooth`
- `decode_single_lane(..., smooth=self.decode_smooth)` kullanılıyor
- Default: `smooth=False` (evaluation için daha güvenli)

### 3) V26 test config’ine smoothing kapalı eklendi
Dosya: `configs/lanelm/lanelm_v4_culane_test_v26_train100.py`
- `decode_cfg.smooth=False`

## Beklenen Etki
- **Strict IoU (0.5, 0.75)** metriklerinde yapay düşüşe sebep olabilecek smoothing etkisi elimine edilir.
- “Zigzag” ile mücadele etmek için smoothing gerekiyorsa, bu artık **kontrollü bir knob** (ablation) olur.

## Doğrulama (Önerilen)
- Aynı checkpoint ile iki test:
  - `smooth=False` (eval doğru mu?)
  - `smooth=True` (görsel kalite vs metrik trade-off)


