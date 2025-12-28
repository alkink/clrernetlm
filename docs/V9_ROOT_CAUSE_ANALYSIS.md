# V9: Root Cause Analizi - CLRNet Pseudo Label Başarısız

## Test Sonuçları (Koordinat Düzeltmesi Sonrası)

### Test 1 (overfit-size 1):
- **F1@0.1**: 0.2965 ❌ (düşük)
- **F1@0.5**: 0.0165 ❌ (çok kötü)
- **TP@0.5**: 5, **FP@0.5**: 395, **FN@0.5**: 202

### Test 2 (overfit-size 8):
- **F1@0.1**: 0.4547 ✅ (biraz iyi)
- **F1@0.5**: 0.0461 ❌ (hala çok kötü)
- **TP@0.5**: 14, **FP@0.5**: 386, **FN@0.5**: 193

**Sonuç:** Koordinat dönüşümü düzeltildi ama F1@0.5 hala çok düşük. **Temel sorun başka!**

## PDF'den Kritik Bulgular

### 1. PDF Sayfa 867-871: "(2-kp)" Versiyonu

> "(2-kp) denotes that the holistic lane predicted from CLRNet is given and **two adjacent ground truth keypoints with random shift** at the commencement of each lane are also supplied to enhance model performance."

**KRİTİK:** PDF'de "(2-kp)" versiyonu:
- CLRNet'ten **holistic lane** alıyor (tüm lane)
- **GT'den** 2 keypoint alıyor (random shift ile)
- **CLRNet'ten değil, GT'den keypoint!**

**Şu anki implementasyonumuz:**
- CLRNet'ten 2 keypoint alıyoruz (Lq)
- GT'den kalan keypoint'leri alıyoruz (Lgt)
- **YANLIŞ!** PDF'de GT'den keypoint alınması gerekiyor!

### 2. PDF Sayfa 879-885: "Analysis on Limitations"

> "(1) In the * version, LaneLM underperforms CLRNet because, in Eq. 10, LaneLM actually predict pseudo-labels from CLRNet i.e. the knowledge of this part in LaneLM is **distilled from the CLRNet**."

**Sorun:** Model CLRNet'in hatalarını öğreniyor. CLRNet yanlış keypoint verirse, model bunu öğreniyor.

> "(2) LaneLM with fewer keypoint prompts is worse than the * version because, in the training sequence, a **sudden jump occurs at the junction between the pseudo-label and the ground truth** (see Eq. 10), which disrupts the contextual semantic information and confuses the model."

**Sorun:** Lq ve Lgt arasındaki "sudden jump" problemi devam ediyor. Noise eklemek yeterli değil.

### 3. PDF Sayfa 1619-1623: "Analysis on hallucination"

> "Eq. 10 endows the model with the capability of VQA but it makes it easier for the model to predict cyclic sequences. Figure 6(a) illustrates that the model has **learned the abrupt change points that connecting Lq and Lgt on the side**. LaneLM has learned the contextual representation of abrupt change points and consequently results in hallucination."

**Sorun:** Model Lq→Lgt geçişindeki "abrupt change points" pattern'ini öğreniyor. Bu, test'te de ortaya çıkıyor → zigzagging.

### 4. PDF Sayfa 887-891: LLAMAS Strategy

> "The training strategy is slightly different with CULane and TuSimple. We directly use Lgt as self-supervised label S and **Lq is not used during training**, which is different with Eq. 10. Thus, we can avoid knowledge distilling from the teacher model."

**LLAMAS'da:** Lq kullanılmıyor, direkt Lgt ile train ediliyor. Bu, CLRNet'in hatalarını öğrenmeyi önler.

## Root Cause Analizi

### Sorun 1: CLRNet Keypoint'leri Yanlış Olabilir

**Training'de:**
- CLRNet'ten 2 keypoint alıyoruz (Lq)
- CLRNet yanlış keypoint verirse → Model bunu öğreniyor
- Bipartite matching yanlış olabilir → Yanlış eşleştirme

**Test'te:**
- CLRNet'ten 2 keypoint alıyoruz (prompting)
- CLRNet yanlış keypoint verirse → Model yanlış başlangıç noktasından başlıyor

**Sonuç:** Model CLRNet'in hatalarını öğreniyor ve test'te de aynı hataları yapıyor.

### Sorun 2: "Sudden Jump" Problemi

**Training'de:**
- Lq (CLRNet'ten) → Lgt (GT'den) geçişi
- Bu geçişte geometrik süreksizlik var
- Noise eklemek yeterli değil

**Test'te:**
- Lq (CLRNet'ten) → Model tahmini geçişi
- Model training'de öğrendiği "sudden jump" pattern'ini uyguluyor → zigzagging

**Sonuç:** Model "abrupt change points" pattern'ini öğreniyor.

### Sorun 3: PDF'nin "(2-kp)" Stratejisi Farklı

**PDF'de "(2-kp)" versiyonu:**
- CLRNet'ten **holistic lane** alıyor (tüm lane, sadece keypoint değil)
- **GT'den** 2 keypoint alıyor (random shift ile)
- Bu, CLRNet'in hatalarını öğrenmeyi önler

**Şu anki implementasyonumuz:**
- CLRNet'ten 2 keypoint alıyoruz (Lq)
- GT'den kalan keypoint'leri alıyoruz (Lgt)
- **YANLIŞ!** PDF'de GT'den keypoint alınması gerekiyor!

## Çözüm: PDF'nin "(2-kp)" Stratejisini Uygula

### PDF'den Alıntı (Sayfa 867-871):

> "(2-kp) denotes that the holistic lane predicted from CLRNet is given and **two adjacent ground truth keypoints with random shift** at the commencement of each lane are also supplied to enhance model performance."

### Implementasyon:

1. **Training'de:**
   - CLRNet'ten holistic lane al (opsiyonel, sadece visual guidance için)
   - **GT'den** ilk 2 keypoint al (Lq)
   - GT'den kalan keypoint'leri al (Lgt)
   - Lq keypoint'lerine random shift ekle (-5 to +5 pixels)
   - Lq ◦ Lgt formatında train et

2. **Test'te:**
   - CLRNet'ten 2 keypoint al (prompting)
   - Model bunları kullanarak devam eder

**Avantajlar:**
- Model CLRNet'in hatalarını öğrenmez (GT'den keypoint)
- "Sudden jump" problemi azalır (GT'den keypoint, daha smooth)
- Training/test uyumsuzluğu olur ama bu kabul edilebilir (PDF'de de var)

**Dezavantajlar:**
- Training/test uyumsuzluğu (Training: GT keypoint, Test: CLRNet keypoint)
- Ama PDF'de de bu var ve çalışıyor

## Alternatif: LLAMAS Strategy (Daha Basit)

PDF sayfa 887-891'de LLAMAS strategy var:
- Training'de Lq kullanma, direkt Lgt ile train et
- Test'te CLRNet prompting kullan

**Avantajlar:**
- Basit, hızlı
- CLRNet'in hatalarını öğrenmez
- "Sudden jump" problemi yok

**Dezavantajlar:**
- Training/test uyumsuzluğu (Training: Lq yok, Test: CLRNet Lq)
- Model CLRNet keypoint'lerini yorumlamayı öğrenmez

## Önerilen Çözüm

**PDF'nin "(2-kp)" stratejisini uygula:**

1. **Training'de:**
   - GT'den ilk 2 keypoint al (Lq)
   - GT'den kalan keypoint'leri al (Lgt)
   - Lq keypoint'lerine random shift ekle (-5 to +5 pixels)
   - Lq ◦ Lgt formatında train et

2. **Test'te:**
   - CLRNet'ten 2 keypoint al (prompting)
   - Model bunları kullanarak devam eder

**Bu, PDF'deki "(2-kp)" stratejisine en yakın ve CLRNet'in hatalarını öğrenmeyi önler.**

## Notlar

- PDF'de "(2-kp)" versiyonu GT'den keypoint alıyor, CLRNet'ten değil
- CLRNet'in hatalarını öğrenme problemi çok kritik
- "Sudden jump" problemi devam ediyor
- LLAMAS strategy alternatif ama training/test uyumsuzluğu var








