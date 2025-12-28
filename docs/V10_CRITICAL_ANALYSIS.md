# V10: Kritik Analiz - Training/Test Uyumsuzluğu

## Test Sonuçları (V10 GT Keypoint Stratejisi Sonrası)

### Test Sonuçları:
- **F1@0.1**: 0.2965 ❌ (düşük)
- **F1@0.5**: 0.0165 ❌ (çok kötü, V9 ile aynı!)
- **TP@0.5**: 5, **FP@0.5**: 395, **FN@0.5**: 202

**Sonuç:** V10 değişikliği (GT keypoint) hiçbir iyileşme sağlamadı. **Temel sorun başka!**

## Root Cause Analizi

### Sorun 1: Training/Test Uyumsuzluğu (Kritik!)

**V10 Training:**
- GT'den ilk 2 keypoint alıyoruz (Lq) - random shift ile
- GT'den kalan keypoint'leri alıyoruz (Lgt)
- Model GT keypoint'lerini öğreniyor

**V10 Test:**
- CLRNet'ten 2 keypoint alıyoruz (prompting)
- Model CLRNet keypoint'lerini yorumlamaya çalışıyor

**Sorun:** Model training'de GT keypoint görüyor, test'te CLRNet keypoint görüyor. **Farklı keypoint formatları!**

### PDF'den Kritik Bulgular

#### 1. PDF Sayfa 887-891: LLAMAS Strategy

> "The training strategy is slightly different with CULane and TuSimple. We directly use Lgt as self-supervised label S and **Lq is not used during training**, which is different with Eq. 10. Thus, we can avoid knowledge distilling from the teacher model."

**LLAMAS'da:**
- Training'de Lq kullanılmıyor
- Direkt Lgt ile train ediliyor
- Test'te CLRNet prompting kullanılıyor
- **Training/test uyumsuzluğu yok!** (Her ikisinde de Lq yok)

#### 2. PDF Sayfa 497-508: Prompting Strategy

> "(1) A regression network is employed to provide the two initial keypoints, for each lane. LaneLM is responsible for completing the remaining keypoints. The regression network (we use CLRNet [6]) only gives start points for each lane rather than the holistic lane..."

**Test'te:**
- CLRNet'ten 2 keypoint alınıyor
- Model bunları kullanarak devam ediyor

#### 3. PDF Sayfa 879-885: "Analysis on Limitations"

> "(2) LaneLM with fewer keypoint prompts is worse than the * version because, in the training sequence, a **sudden jump occurs at the junction between the pseudo-label and the ground truth** (see Eq. 10), which disrupts the contextual semantic information and confuses the model."

**Sorun:** Lq ve Lgt arasındaki "sudden jump" problemi devam ediyor.

## Çözüm: LLAMAS Strategy (PDF'den)

### PDF'den Alıntı (Sayfa 887-891):

> "We directly use Lgt as self-supervised label S and **Lq is not used during training**, which is different with Eq. 10. Thus, we can avoid knowledge distilling from the teacher model."

### Implementasyon:

1. **Training'de:**
   - Lq kullanma (ne CLRNet'ten, ne GT'den)
   - Direkt GT keypoint'lerini al (Lgt)
   - Lgt formatında train et (Lq ◦ Lgt değil, sadece Lgt)

2. **Test'te:**
   - CLRNet'ten 2 keypoint al (prompting)
   - Model bunları kullanarak devam eder

**Avantajlar:**
- Training/test uyumsuzluğu yok (Her ikisinde de Lq yok training'de)
- CLRNet'in hatalarını öğrenmez (Lq kullanılmıyor)
- "Sudden jump" problemi yok (Lq yok, direkt Lgt)
- PDF'de LLAMAS'da çalışıyor

**Dezavantajlar:**
- Model CLRNet keypoint'lerini yorumlamayı öğrenmez (ama test'te kullanıyoruz)
- Ama PDF'de LLAMAS'da bu çalışıyor!

## Alternatif: PDF'nin "*" Versiyonu (CLRNet Lq + Bipartite Matching)

PDF sayfa 867-871'de "*" versiyonu var:
- Training: CLRNet Lq ◦ GT Lgt (bipartite matching)
- Test: CLRNet prompting (Lq from CLRNet)
- **Training/test uyumlu!** (Her ikisinde de CLRNet Lq)

**Ama PDF'de "*" versiyonu CLRNet'ten daha kötü performans gösteriyor:**
> "(1) In the * version, LaneLM underperforms CLRNet because, in Eq. 10, LaneLM actually predict pseudo-labels from CLRNet i.e. the knowledge of this part in LaneLM is **distilled from the CLRNet**."

## Önerilen Çözüm

**LLAMAS Strategy'yi uygula:**

1. **Training'de:**
   - Lq kullanma (ne CLRNet'ten, ne GT'den)
   - Direkt GT keypoint'lerini al (Lgt)
   - Lgt formatında train et (Lq ◦ Lgt değil, sadece Lgt)

2. **Test'te:**
   - CLRNet'ten 2 keypoint al (prompting)
   - Model bunları kullanarak devam eder

**Bu, PDF'deki LLAMAS strategy'ye uygun ve training/test uyumsuzluğunu önler.**

## Notlar

- V10 (GT keypoint) hiçbir iyileşme sağlamadı
- Training/test uyumsuzluğu kritik sorun
- LLAMAS strategy PDF'de çalışıyor
- "*" versiyonu CLRNet'ten kötü (knowledge distillation problemi)






