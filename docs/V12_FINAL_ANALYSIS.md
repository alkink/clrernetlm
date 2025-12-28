# V12: Final Analysis - "Abrupt Change Points" Problemi

## Test Sonuçları

- **F1@0.5:** 0.0000-0.0626 (çok kötü)
- **Kullanıcı:** "yolları da çok iyi coverlamıyor" - Geometrik olarak da yanlış tahmin ediyor
- **Training Loss:** 0.0050 (çok iyi, overfitting)

## Root Cause: PDF'den Kritik Bulgu

### PDF Sayfa 879-885:

> "Analysis on Limitations. (1) In the * version, LaneLM underperforms CLRNet because, in Eq. 10,
> LaneLM actually predict pseudo-labels from CLRNet i.e. the knowledge of this part in LaneLM is
> distilled from the CLRNet. **(2) LaneLM with fewer keypoint prompts is worse than the * version
> because, in the training sequence, a sudden jump occurs at the junction between the pseudo-label and
> the ground truth (see Eq. 10), which disrupts the contextual semantic information and confuses the
> model. It has been observed that the model often hallucinates on the side lanes, indicating that the
> model struggles to cope with abrupt changes in semantic information.**"

### PDF Sayfa 1619-1623:

> "Analysis on hallucination. Current large language models are still struggling with hallucination.
> Figure 6(a) shows hallucination in LaneLM. Eq. 10 endows the model with the capability of VQA but
> it makes it easier for the model to predict cyclic sequences. **Figure 6(a) illustrates that the model has
> learned the abrupt change points that connecting Lq and Lgt on the side. LaneLM has learned the
> contextual representation of abrupt change points and consequently results in hallucination.**"

## Sorun Analizi

### Training'de Ne Oluyor?

**Training Sequence (Lq ◦ Lgt):**
```
x_tokens = [Lq_kp1, Lq_kp2, Lgt_kp1, Lgt_kp2, Lgt_kp3, ...]
           [  0   ,   1   ,    2    ,    3    ,    4    , ...]
```

**x_in (Shift Right):**
```
x_in = [Lq_kp1, Lq_kp1, Lq_kp2, Lgt_kp1, Lgt_kp2, ...]
       [  0   ,   1   ,    2   ,    3   ,    4   , ...]
```

**Model Forward:**
- Model tüm sequence'i paralel işliyor
- x_in[2] = Lq_kp2 (Lq'nun ikinci keypoint'i)
- x_in[3] = Lgt_kp1 (Lgt'nin ilk keypoint'i)
- **ABRUPT CHANGE!** Lq_kp2 → Lgt_kp1 arasında büyük bir fark var!

**Loss:**
- Loss sadece Lgt kısmında hesaplanıyor (loss_mask)
- Model x_in[3]'te Lgt_kp1'i görüyor ama x_in[2]'de Lq_kp2 var
- Model bu "abrupt change" pattern'ini öğreniyor!

### Test'te Ne Oluyor?

**Test Sequence (CLRNet Prompting):**
```
x_out = [CLR_kp1, CLR_kp2, pred_kp1, pred_kp2, pred_kp3, ...]
        [   0   ,    1   ,     2   ,     3   ,     4   , ...]
```

**x_in (Shift Right):**
```
x_in = [CLR_kp1, CLR_kp1, CLR_kp2, pred_kp1, pred_kp2, ...]
       [   0   ,    1   ,    2   ,     3   ,     4   , ...]
```

**Model Forward:**
- Model tüm sequence'i paralel işliyor
- x_in[2] = CLR_kp2 (CLRNet'in ikinci keypoint'i)
- x_in[3] = pred_kp1 (Model'in ilk prediction'ı)
- **ABRUPT CHANGE!** CLR_kp2 → pred_kp1 arasında büyük bir fark olabilir!

**Sorun:** Model training'de Lq → Lgt "abrupt change" öğreniyor, test'te de CLR → pred "abrupt change" yapıyor!

## Çözüm: Training'i Değiştir - "Abrupt Change" Problemini Azalt

### Seçenek 1: LLAMAS Stratejisi (PDF Sayfa 887-891)

PDF'de LLAMAS için farklı bir strateji var:
> "Performance on LLAMAS. The result on the LLAMAS is shown in Table 4. LaneLM-512 outperforms PolyLaneNet [20] by 7.05 and LaneATT [8] by 2.08. **The training strategy is slightly different with CULane and TuSimple. We directly use Lgt as self-supervised label S and Lq is not used during training, which is different with Eq. 10. Thus, we can avoid knowledge distilling from the teacher model.**"

**Ama bu CULane için değil, LLAMAS için!** PDF'de CULane için "*" version kullanılıyor.

### Seçenek 2: Noise'u Artır (PDF Section 3.4)

PDF'de noise ekleniyor ama belki yeterli değil. Noise'u artırabiliriz:
- Şu an: -5 to +5 pixels
- Öneri: -10 to +10 pixels veya daha fazla

### Seçenek 3: Training'de Lq'nun Tüm Keypoint'lerini Kullan (PDF "*" Version)

PDF'de "*" version şöyle:
> "Performance on CULane. Table 3 reports the main results on CULane. **Our model receives two adjacent keypoints output from CLRNet [6] as init prompts for each lane and rollouts the remaining keypoints in the * version.**"

**Ama biz zaten bunu yapıyoruz!** Training'de CLRNet'ten 2 keypoint alıyoruz (Lq), test'te de 2 keypoint alıyoruz.

### Seçenek 4: Training'de Loss'u Lq Kısmında da Hesapla (Ama Düşük Weight)

PDF'de loss sadece Lgt kısmında hesaplanıyor. Belki Lq kısmında da loss hesaplayabiliriz ama düşük weight ile:
- Lq loss weight: 0.1
- Lgt loss weight: 1.0

Bu model'e Lq'nun da önemli olduğunu öğretir.

### Seçenek 5: Training'de Lq ve Lgt Arasında Smooth Transition

Lq ve Lgt arasında smooth transition oluşturabiliriz:
- Lq'nun son keypoint'i ile Lgt'nin ilk keypoint'i arasında interpolasyon
- Veya Lq'nun son keypoint'ini Lgt'nin ilk keypoint'ine yaklaştır

## Önerilen Çözüm

**Seçenek 4 + Seçenek 2 Kombinasyonu:**
1. **Noise'u artır:** -5 to +5 → -10 to +10 pixels
2. **Lq loss ekle:** Lq kısmında da loss hesapla ama düşük weight (0.1) ile
3. **Smooth transition:** Lq'nun son keypoint'i ile Lgt'nin ilk keypoint'i arasında interpolasyon

Bu "abrupt change" problemini azaltır ve model'in daha smooth öğrenmesini sağlar.

## Notlar

- PDF'de "abrupt change points" problemi açıkça belirtilmiş
- Model Lq ve Lgt arasındaki "abrupt change" öğreniyor
- Training'de loss sadece Lgt kısmında hesaplanıyor
- Test'te de CLR → pred "abrupt change" var
- Çözüm: Training'i değiştir, "abrupt change" problemini azalt






