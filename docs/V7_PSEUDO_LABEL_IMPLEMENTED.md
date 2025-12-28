# V7: Pseudo Label Training Strategy Implemented

## Problem
Model prompting strategy'yi öğrenemiyordu çünkü training'de pseudo label kullanılmıyordu. PDF'deki Equation 10 formatı (`Lq ◦ Lgt`) yoktu.

## Solution
Training'e pseudo label stratejisi eklendi:

### 1. Lq ◦ Lgt Formatı
- **Lq (Pseudo Label)**: GT'den ilk 2 keypoint alınıyor (prompt olarak)
- **Lgt (Ground Truth)**: Kalan keypoint'ler (answer olarak)
- **Sequence**: `Lq ◦ Lgt` formatında concatenate ediliyor

### 2. Loss Masking
- Loss sadece **Lgt kısmı** için hesaplanıyor
- **Lq kısmı** input olarak kullanılıyor, loss yok
- Model Lq'yu görüp Lgt'yi tahmin etmeye çalışıyor

### 3. Implementation Details
- `num_pseudo_points = 2`: PDF'deki gibi ilk 2 keypoint
- `loss_mask`: Lq pozisyonlarında False, Lgt pozisyonlarında True
- X-loss, Y-loss ve AR-loss hepsi Lgt kısmı için hesaplanıyor

## Expected Impact
- Model prompting strategy'yi öğrenecek
- Test'te CLRNet'ten gelen initial keypoint'leri kullanabilecek
- F1@0.5 skoru artmalı (şu an 0.0165, hedef: >0.5)

## Next Steps
1. Model'i yeniden train et (pseudo label ile)
2. Test et: Model prompting strategy kullanabilmeli
3. İleride: CLRNet'ten gerçek pseudo label al (şu an GT'den alınıyor)

## Notes
- Şu an GT'den ilk 2 keypoint alınıyor (basit yaklaşım)
- PDF'de CLRNet'ten pseudo label alınıyor, ama bu daha karmaşık
- Bu yaklaşım model'in prompting strategy'yi öğrenmesine yardımcı olacak








