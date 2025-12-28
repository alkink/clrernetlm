# V9: Kritik Analiz - CLRNet Pseudo Label Başarısız

## Test Sonuçları

### Test 1 (overfit-size 1):
- **F1@0.1**: 0.4613 ✅ (iyi - model lane'leri bulabiliyor)
- **F1@0.5**: 0.0494 ❌ (çok kötü - geometrik hizalama sorunu)
- **F1@0.75**: 0.0000 ❌ (felaket)
- **TP@0.5**: 15, **FP@0.5**: 385, **FN@0.5**: 192

### Test 2 (overfit-size 8):
- **F1@0.1**: 0.3690 ✅ (biraz düşük ama hala iyi)
- **F1@0.5**: 0.0033 ❌ (daha da kötü!)
- **F1@0.75**: 0.0000 ❌ (felaket)
- **TP@0.5**: 1, **FP@0.5**: 399, **FN@0.5**: 206

## Sorun Analizi

### 1. **CLRNet Keypoint Formatı Hatası (KRİTİK!)**

**Training'de:**
```python
# CLRNet'ten gelen keypoint'ler normalized [0,1] (ori_img_w, ori_img_h bazında)
first_2_clr = clr_points[:num_pseudo_points].copy()  # (2, 2) normalized [0,1]

# Yanlış dönüşüm:
x_resized = first_2_clr[:, 0] * (crop_bbox[2] - crop_bbox[0])  # Denormalize
y_resized = first_2_clr[:, 1] * (crop_bbox[3] - crop_bbox[1])
```

**Sorun:**
- CLRNet'in `points` özelliği **zaten normalized [0,1] koordinatlar** (ori_img_w=1640, ori_img_h=590 bazında)
- Ama training kodunda `(crop_bbox[2] - crop_bbox[0])` ile çarpıyoruz, bu yanlış!
- Doğru dönüşüm: `x_resized = first_2_clr[:, 0] * ori_img_w` (1640)

**Test'te (LaneLMDetector):**
```python
# Test'te doğru dönüşüm:
x_resized = first_2_points[:, 0] * self.ori_img_w  # Denormalize X (1640)
y_resized = first_2_points[:, 1] * self.ori_img_h  # Denormalize Y (590)
```

**Sonuç:** Training ve test'te farklı koordinat sistemleri kullanılıyor! Bu, training/test uyumsuzluğuna neden oluyor.

### 2. **PDF'den Kritik Nokta (Sayfa 879-885)**

> "(1) In the * version, LaneLM underperforms CLRNet because, in Eq. 10, LaneLM actually predict pseudo-labels from CLRNet i.e. the knowledge of this part in LaneLM is distilled from the CLRNet."

**Sorun:** Model CLRNet'in hatalarını öğreniyor. CLRNet yanlış keypoint verirse, model bunu öğreniyor.

> "(2) LaneLM with fewer keypoint prompts is worse than the * version because, in the training sequence, a sudden jump occurs at the junction between the pseudo-label and the ground truth (see Eq. 10), which disrupts the contextual semantic information and confuses the model."

**Sorun:** Lq ve Lgt arasındaki "sudden jump" problemi devam ediyor. Noise eklemek yeterli değil.

### 3. **PDF'den Hallucination Analizi (Sayfa 1619-1623)**

> "Eq. 10 endows the model with the capability of VQA but it makes it easier for the model to predict cyclic sequences. Figure 6(a) illustrates that the model has learned the abrupt change points that connecting Lq and Lgt on the side."

**Sorun:** Model Lq→Lgt geçişindeki "abrupt change points" pattern'ini öğreniyor. Bu, test'te de ortaya çıkıyor → zigzagging.

### 4. **Bipartite Matching Sorunu**

**Training'de:**
- Start point distance kullanıyoruz
- Ama CLRNet keypoint'leri yanlış koordinat sisteminde → yanlış matching

**Test'te:**
- CLRNet keypoint'leri doğru koordinat sisteminde
- Ama training'de yanlış öğrenmiş → test'te çalışmıyor

### 5. **Koordinat Sistemi Uyumsuzluğu**

**Training:**
- CLRNet keypoint'leri: normalized [0,1] (ori_img_w=1640, ori_img_h=590)
- Yanlış dönüşüm: `* (crop_bbox[2] - crop_bbox[0])` = `* 1640` (yanlış!)
- Doğru dönüşüm: `* ori_img_w` = `* 1640` (ama sonra crop/resize gerekir)

**Test:**
- CLRNet keypoint'leri: normalized [0,1] (ori_img_w=1640, ori_img_h=590)
- Doğru dönüşüm: `* ori_img_w` = `* 1640`, sonra crop/resize

**Sonuç:** Training ve test'te farklı koordinat dönüşümleri → Model yanlış öğreniyor!

## Root Cause

**Temel Sorun:** Training'de CLRNet keypoint'lerinin koordinat dönüşümü yanlış!

1. CLRNet `points` özelliği normalized [0,1] (ori_img_w=1640, ori_img_h=590 bazında)
2. Training'de yanlış dönüşüm: `* (crop_bbox[2] - crop_bbox[0])` = `* 1640` (bu doğru ama sonra crop/resize eksik)
3. Test'te doğru dönüşüm: `* ori_img_w` = `* 1640`, sonra crop/resize
4. **Sonuç:** Training ve test'te farklı koordinat sistemleri → Model yanlış öğreniyor!

## Çözüm

### 1. **Koordinat Dönüşümünü Düzelt (KRİTİK!) - ✅ DÜZELTİLDİ**

Training'de CLRNet keypoint'lerinin dönüşümünü test ile aynı yaptım:

```python
# CLRNet keypoint'leri normalized [0,1] (ori_img_w=1640, ori_img_h=590 bazında)
first_2_clr = clr_points[:num_pseudo_points].copy()  # (2, 2) normalized [0,1]

# Doğru dönüşüm (test ile aynı):
ori_img_w = crop_bbox[2] - crop_bbox[0]  # 1640
ori_img_h = crop_bbox[3] - crop_bbox[1]  # 590

# 1. Denormalize to original image space
x_resized = first_2_clr[:, 0] * ori_img_w  # 1640
y_resized = first_2_clr[:, 1] * ori_img_h  # 590

# 2. Apply crop and resize (same as test)
x_min, y_min, x_max, y_max = crop_bbox
x_resized = x_resized - x_min
y_resized = y_resized - y_min
x_scale = img_scale[0] / (x_max - x_min)
y_scale = img_scale[1] / (y_max - y_min)
x_resized = x_resized * x_scale
y_resized = y_resized * y_scale
```

**Değişiklikler:**
- Bipartite matching için start point dönüşümü düzeltildi
- Lq keypoint dönüşümü düzeltildi
- Noise ekleme pixel space'de yapılıyor (daha doğru)

### 2. **PDF'nin Önerdiği Stratejiyi Tam Uygula**

PDF'de "(2-kp)" versiyonu var:
> "two adjacent ground truth keypoints with random shift at the commencement of each lane are also supplied to enhance model performance"

**Bu, GT'den keypoint almak anlamına geliyor, CLRNet'ten değil!**

PDF'de "* version" CLRNet kullanıyor ama "(2-kp)" versiyonu GT kullanıyor.

### 3. **Alternatif: Hybrid Strategy**

- %50 batch: CLRNet Lq (doğru koordinat dönüşümü ile)
- %50 batch: GT Lq (noise ile)

Bu, model'in hem CLRNet hem de GT keypoint'lerini öğrenmesini sağlar.

## Önerilen Çözüm

**1. Koordinat dönüşümünü düzelt (KRİTİK!)**
- Training'de CLRNet keypoint'lerinin dönüşümünü test ile aynı yap
- `ori_img_w` ve `ori_img_h` kullan (1640, 590)

**2. Test et**
- 1-image overfit
- 8-image overfit
- 100-image test

**3. Eğer hala sorun varsa:**
- PDF'nin "(2-kp)" stratejisini uygula (GT'den keypoint, noise ile)
- Veya hybrid strategy (CLRNet + GT)

## Notlar

- Koordinat dönüşümü hatası çok kritik
- Training ve test'te farklı koordinat sistemleri → Model yanlış öğreniyor
- PDF'de "(2-kp)" versiyonu GT kullanıyor, CLRNet değil
- "Sudden jump" problemi devam ediyor → Noise yeterli değil

