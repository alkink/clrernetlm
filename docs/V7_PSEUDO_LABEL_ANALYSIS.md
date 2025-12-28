# V7: Pseudo Label Training - Detaylı Analiz ve Sorunlar

## Test Sonuçları (20251206_020819)
- **F1@0.1**: 0.3427 (önceden 0.5041) - **DÜŞTÜ**
- **F1@0.5**: 0.0033 (önceden 0.0165) - **ÇOK DÜŞÜK, DÜŞTÜ**
- **F1@0.75**: 0.0000 (önceden 0.0000) - **Aynı**

## Kritik Sorun: Training/Test Uyumsuzluğu

### 1. Training'de Ne Oluyor?
- **Lq (Pseudo Label)**: GT'den ilk 2 keypoint alınıyor
- **Lgt (Ground Truth)**: Kalan keypoint'ler
- **Sequence**: `Lq ◦ Lgt` formatında concatenate

### 2. Test'te Ne Oluyor?
- **Lq (Pseudo Label)**: CLRNet'ten ilk 2 keypoint alınıyor
- **Lgt (Ground Truth)**: Model'in tahmin etmesi gereken kalan keypoint'ler

### 3. UYUMSUZLUK!
- Training'de GT'den Lq alıyoruz → Model GT keypoint'leri görmeyi öğreniyor
- Test'te CLRNet'ten Lq alıyoruz → Model CLRNet keypoint'lerini görmeyi öğrenmemiş!

## PDF'den Kritik Bulgular

### Equation 10 (PDF sayfa 7):
```
S = (L1_q ◦ L1_gt, ..., LN_q ◦ LN_gt)
```
- **Lq**: Pseudo labels from CLRNet (teacher model)
- **Lgt**: Ground truth labels
- **Training**: Model Lq'yu görüp Lgt'yi tahmin etmeye çalışıyor

### PDF Sayfa 12 (Analysis on Limitations):
> "(1) In the * version, LaneLM underperforms CLRNet because, in Eq. 10, LaneLM actually predict pseudo-labels from CLRNet i.e. the knowledge of this part in LaneLM is distilled from the CLRNet."

> "(2) LaneLM with fewer keypoint prompts is worse than the * version because, in the training sequence, a sudden jump occurs at the junction between the pseudo-label and the ground truth (see Eq. 10), which disrupts the contextual semantic information and confuses the model."

### Kritik Nokta:
- **"Sudden jump"**: Lq ve Lgt arasında ani bir sıçrama var
- Bu sıçrama model'i karıştırıyor
- Model "abrupt change points" öğreniyor ve bu hallucination'a yol açıyor

## Mevcut Implementation Sorunları

### 1. Training/Test Uyumsuzluğu
- **Training**: GT'den Lq (ilk 2 keypoint)
- **Test**: CLRNet'ten Lq (ilk 2 keypoint)
- **Sonuç**: Model CLRNet keypoint'lerini görmeyi öğrenmemiş!

### 2. "Sudden Jump" Problemi
- Lq (ilk 2 keypoint) ve Lgt (kalan keypoint'ler) arasında geometrik süreksizlik olabilir
- Model bu süreksizliği öğreniyor ve test'te hallucination yapıyor

### 3. Loss Masking Sorunu
- Loss sadece Lgt kısmı için hesaplanıyor
- Ama Lq kısmı da model'in input'u
- Model Lq'yu görüyor ama loss yok → Model Lq'yu ignore edebilir

## Çözüm Önerileri

### 1. ✅ Training'e CLRNet Noise Simulation Eklendi
- **Uygulandı**: Lq keypoint'lerine random noise (-5 to +5 pixels) eklendi
- **PDF Section 3.4**: "randomly shifting the x-coordinates by -5 to 5 pixels"
- **Amaç**: CLRNet'in hatalarını simüle etmek ve "sudden jump" problemini azaltmak
- **Sonuç**: Model süreksizliği daha iyi handle edebilecek

### 2. ⚠️ Training'e CLRNet Pseudo Label Ekle (HENÜZ YAPILMADI)
- Training'de CLRNet'ten gerçek pseudo label almak daha iyi olur
- Ama bu training'i çok yavaşlatır (her batch için CLRNet inference)
- Şu an noise simulation ile simüle ediyoruz
- **İleride**: Gerçek CLRNet pseudo label eklenebilir

### 3. Loss Masking'i Gözden Geçir
- Şu an Lq kısmı için loss yok
- Ama model Lq'yu görüyor
- Belki Lq kısmı için de küçük bir loss eklenebilir (auxiliary loss)

### 4. Training Strategy Değiştir
- PDF'de "2-kp" versiyonu var: "two adjacent ground truth keypoints with random shift"
- Bu "sudden jump" problemini azaltır
- Training'de GT keypoint'lerine random shift ekle

## Sonraki Adımlar

1. **Training'e CLRNet Pseudo Label Ekle** (en kritik!)
   - CLRNet model'ini training'de kullan
   - Her batch için CLRNet'ten pseudo label al
   - Lq olarak CLRNet keypoint'lerini kullan

2. **"Sudden Jump" Problemini Azalt**
   - Lq keypoint'lerine random noise ekle (-5 to +5 pixels)
   - Bu training augmentation olarak

3. **Test ve Analiz**
   - Model'i yeniden train et
   - Test et ve sonuçları karşılaştır

