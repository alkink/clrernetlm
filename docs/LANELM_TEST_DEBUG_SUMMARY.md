# LaneLM Test Class Debug Özeti

## Test Class: LaneLMDetector

### Dosya Konumu
`libs/models/detectors/lanelm_detector.py`

### Amaç
LaneLM modelini MMEngine framework'ü ile uyumlu hale getirmek ve CULane dataset'i üzerinde resmi test yapabilmek.

### Yapı
- `BaseDetector` sınıfından türetilmiş
- CLRerNet backbone ve neck'i kullanıyor
- LaneLM modelini wrap ediyor
- `predict()` metodu ile inference yapıyor
- `extract_feat()` metodu ile FPN feature'ları çıkarıyor

## Yaşadığımız Sorunlar ve Çözüm Denemeleri

### 1. Model Ağırlıklarının Yüklenmemesi

**Sorun:** MMEngine'in `load_from` mekanizması LaneLM ağırlıklarını override ediyordu.

**Çözüm Denemesi:**
- `_load_from_state_dict()` metodunu override ettik
- CLRerNet ağırlıkları yüklendikten sonra LaneLM ağırlıklarını manuel olarak yüklüyoruz

**Durum:** ✅ Çözüldü (ağırlıklar yükleniyor)

### 2. Spline Interpolation Hatası

**Sorun:** `ValueError: x must be increasing if s > 0` - Lane objesi oluşturulurken Y koordinatları artan sırada değildi.

**Çözüm Denemesi:**
- `coords_to_lane_normalized()` fonksiyonunda noktaları Y'ye göre sıraladık
- Aynı Y değerlerine sahip noktaları temizledik
- Y'nin kesinlikle artan olduğunu garanti ettik

**Durum:** ✅ Çözüldü (spline hatası yok)

### 3. Normalizasyon Hataları

**Sorun:** X ve Y koordinatlarının normalizasyonunda hatalar vardı.

**Çözüm Denemeleri:**
- `coords_to_lane_normalized()` içinde:
  - X normalizasyonu: `xs / img_w` (önceden `xs / (img_w - 1)` idi)
  - Y normalizasyonu: `y_full / ori_img_h` (önceden `y_full / (ori_img_h - 1)` idi)

**Durum:** ✅ Düzeltildi (normalizasyon formülleri doğru)

### 4. get_prediction_string Extrapolation Sorunu

**Sorun:** `get_prediction_string()` fonksiyonu lane'in Y range'i dışındaki değerler için extrapolation yapıyordu, bu da yanlış X koordinatlarına yol açıyordu.

**Çözüm Denemesi:**
- `get_prediction_string()` içinde Y değerlerini lane'in `min_y` ve `max_y` aralığına filtreledik
- Sadece lane'in geçerli Y aralığı içindeki değerler için interpolation yapıyoruz

**Durum:** ✅ Düzeltildi (extrapolation önlendi)

### 5. Görüntü Normalizasyonu Eksikliği

**Sorun:** `LaneLMDetector.predict()` içinde görüntüler [0, 255] aralığında kalıyordu, model ise [0, 1] aralığında eğitilmişti.

**Çözüm Denemesi:**
- `predict()` metodunda görüntü normalizasyonu eklendi:
  ```python
  if imgs.dtype == torch.uint8:
      imgs = imgs.float() / 255.0
  elif imgs.max() > 1.0:
      imgs = imgs / 255.0
  ```

**Durum:** ✅ Düzeltildi (görüntüler normalize ediliyor)

### 6. Prediction Dosyası Formatı

**Sorun:** Prediction dosyaları doğru formatta oluşturulmuyordu.

**Çözüm Denemesi:**
- `CULaneMetric.compute_metrics()` içinde prediction dosyaları doğru formatta kaydediliyor
- `result_dir` parametresi ile prediction dosyaları kalıcı olarak kaydediliyor

**Durum:** ✅ Çözüldü (prediction dosyaları doğru formatta)

## Mevcut Durum

### Test Sonuçları (100 görüntü, test_valid_gt_100.txt)

**ÖNCEKİ KOŞU (test_valid_gt_100.txt, eski checkpoint)**  

- **IoU Threshold = 0.1:**
  - Precision: 0.2450
  - Recall: 0.3267
  - F1: 0.2800

- **IoU Threshold = 0.5:**
  - Precision: 0.0000
  - Recall: 0.0000
  - **F1: 0.0000** ❌

- **IoU Threshold = 0.75:**
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000

---

**YENİ KOŞU (2025-12-04, test_100.txt, V6 + 8-image overfit checkpoint)**  

- **Eğitim tarafı:**  
  - `train_lanelm_v4_fixed.py --epochs 200 --overfit-size 8`  
  - 8 görsele **tam overfit**: `best loss ≈ 0.0057` (X≈0.0048, AR≈0.0008, PRES≈1e-4, PIX=0, Y-loss kapalı)
  - Bu, modelin görsel bilgiyi ve token uzayını çok güçlü şekilde ezberleyebildiğini gösteriyor.

- **Test tarafı (`configs/lanelm/lanelm_v4_culane_test.py`, `test_100.txt`):**
  - **IoU Threshold = 0.1:**
    - `test_all`: TP=144, FP=256, FN=63 → Precision=0.3600, Recall=0.6957, **F1=0.4745**
  - **IoU Threshold = 0.5:**
    - `test_all`: TP=4, FP=396, FN=203 → Precision=0.0100, Recall=0.0193, **F1=0.0132**
  - **IoU Threshold = 0.75:**
    - `test_all`: TP=0, FP=400, FN=207 → **F1=0.0000**

- **Yorum:**  
  - 0.1 IoU’da F1’in 0.47’ye çıkması, tahminlerin **doğru şerit bölgesine** kabaca oturduğunu gösteriyor (artık “tamamen alakasız” değiller).  
  - Ancak 0.5 ve 0.75’te F1’in yine neredeyse sıfır olması, **şekil/jeometri uyumsuzluğunun** hâlâ ciddi olduğunu gösteriyor:
    - Şeritler GT ile aynı bölgede ama:
      - X-ekseni boyunca küçük zigzag / jitter,
      - Y boyunca örnekleme farkları ve spline tabanlı lane temsil farkı,
      - 30px mask genişliği altında, bu jitter birikerek IoU’yu hızlıca 0.5 altına düşürüyor.
  - Özetle: **eğitimde aşırı güçlü ezber var (8 görselde mükemmel overfit), ama CULane IoU@0.5 metriği halen “şekil/akışkanlık” açısından cezalandırıyor.**

### Prediction Kalitesi Analizi

Örnek görüntü (`driver_100_30frame/05251517_0433.MP4/00930.jpg`) üzerinde:

**GT Lane 0:**
- X: [-4.3, 663.7]
- Y: [330.0, 490.0]

**Pred Lane 0 (en yakın):**
- X: [31.4, 1399.2]
- Y: [266.0, 586.0]
- Mean X diff: 312.4px
- Median X diff: 232.9px
- Max X diff: 879.0px
- Points < 30px: 0/80 (0.0%)
- Points < 50px: 0/80 (0.0%)

### Sorunun Kök Nedeni

**F1@0.5 = 0.0000** olmasının nedeni:
- Model prediction'ları GT'ye çok uzak (300-700px X hatası)
- IoU 0.5 threshold'u için genellikle 30-50px hata kabul edilebilir
- Mevcut hatalar 10-20 kat daha büyük

**Olası Nedenler:**
1. Model yeterince eğitilmemiş (2000 görüntü, 200 epoch yeterli olmayabilir)
2. Model architecture'ında sorunlar olabilir
3. Training ve test arasında data preprocessing farklılıkları olabilir
4. Model posterior collapse yaşıyor olabilir (görsel bilgiyi kullanmıyor)

## Yapılan Düzeltmeler Özeti

1. ✅ Model ağırlıklarının doğru yüklenmesi
2. ✅ Spline interpolation hatalarının düzeltilmesi
3. ✅ Normalizasyon formüllerinin düzeltilmesi
4. ✅ get_prediction_string extrapolation sorununun çözülmesi
5. ✅ Görüntü normalizasyonunun eklenmesi
6. ✅ Prediction dosyası formatının düzeltilmesi

## Hala Çözülmemiş Sorunlar

1. ❌ **F1@0.5 = 0.0000** - Model prediction'ları GT'ye çok uzak
2. ❌ X koordinat hataları çok büyük (300-700px)
3. ❌ Model performansı yetersiz

## Öneriler

1. **Model Eğitimi:**
   - Daha fazla epoch (200 → 400-600)
   - Daha büyük dataset (2000 → 10000+)
   - Learning rate schedule optimizasyonu

2. **Model Architecture:**
   - Cross-attention mekanizmasının daha iyi çalışması için fine-tuning
   - Visual encoder'ın daha güçlü olması
   - Decoder'ın daha iyi öğrenmesi

3. **Data Preprocessing:**
   - Training ve test arasında preprocessing tutarlılığının kontrolü
   - Augmentation stratejisinin gözden geçirilmesi

4. **Debug:**
   - Model'in görsel bilgiyi kullanıp kullanmadığının kontrolü (cross-attention analizi)
   - Training loss'un daha detaylı analizi
   - Validation set üzerinde detaylı analiz

## Test Class Kod Yapısı

### Ana Metodlar

1. **`__init__()`**: Model, backbone, neck, tokenizer yapılandırması
2. **`_load_lanelm_weights()`**: LaneLM ağırlıklarını yükleme
3. **`extract_feat()`**: FPN feature'larını çıkarma
4. **`predict()`**: Inference yapma ve Lane objeleri oluşturma
5. **`_load_from_state_dict()`**: MMEngine checkpoint yükleme override

### Yardımcı Fonksiyonlar

1. **`autoregressive_decode()`**: Greedy decoding ile token üretme
2. **`coords_to_lane_normalized()`**: Resized koordinatları normalized Lane objesine çevirme

## Config Dosyası

`configs/lanelm/lanelm_v4_culane_test.py`

- CLRerNet config'inden inherit ediyor
- `LaneLMDetector` model olarak tanımlanmış
- Test dataset: `test_valid_gt_100.txt` (100 görüntü)
- Prediction dosyaları: `work_dirs/lanelm_v4_test_valid_gt/predictions/`

## Sonuç

Tüm teknik sorunlar (normalizasyon, format, ağırlık yükleme) çözüldü, ancak **model performansı hala yetersiz**. F1@0.5 = 0.0000 olması, modelin GT'ye yeterince yakın prediction üretemediğini gösteriyor. Bu, model eğitimi veya architecture ile ilgili bir sorun olabilir.


