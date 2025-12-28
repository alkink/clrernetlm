# V7: Pseudo Label Training Fix - Özet

## Yapılan Değişiklikler

### 1. Lq Keypoint'lerine Random Noise Eklendi
- **PDF Section 3.4**: "randomly shifting the x-coordinates by -5 to 5 pixels"
- **Amaç**: CLRNet'in hatalarını simüle etmek ve "sudden jump" problemini azaltmak
- **Implementation**: `lq_noise_range = 5` pixels, her training batch'te random noise ekleniyor

### 2. Lq ve Lgt Ayrımı Düzeltildi
- **Önceki**: GT'den ilk 2 keypoint → Lq, kalan → Lgt (token space'de)
- **Yeni**: GT'den ilk 2 keypoint (pixel space) → noise ekle → Lq, kalan → Lgt
- **Avantaj**: Noise pixel space'de ekleniyor, daha gerçekçi

## Beklenen Etkiler

1. **"Sudden Jump" Problemi Azalmalı**
   - Model Lq ve Lgt arasındaki süreksizliği daha iyi handle edecek
   - Random noise model'i daha robust hale getirecek

2. **Training/Test Uyumu İyileşmeli**
   - Training'de noisy Lq, test'te CLRNet Lq → daha benzer
   - Model CLRNet keypoint'lerini daha iyi kullanabilecek

3. **F1@0.5 Skoru Artmalı**
   - Şu an: 0.0033 (çok düşük)
   - Hedef: >0.1 (minimum), ideal: >0.5

## Sonraki Adımlar

1. **Model'i Yeniden Train Et**
   - `python tools/train_lanelm_v4_fixed.py --overfit-size 1` (1-image overfit test)
   - Loss'un düşüp düşmediğini kontrol et

2. **Test Et**
   - Model'i test et ve F1@0.5 skorunu kontrol et
   - Görselleştirmeleri incele (zigzagging azaldı mı?)

3. **İleride: Gerçek CLRNet Pseudo Label**
   - Training'e CLRNet inference ekle (her batch için)
   - Bipartite matching ekle (PDF'de var)
   - Bu training'i yavaşlatır ama daha doğru olur

## Notlar

- Şu an noise simulation ile CLRNet hatalarını simüle ediyoruz
- Bu basit ve hızlı bir çözüm
- İleride gerçek CLRNet pseudo label eklenebilir (daha karmaşık ama daha doğru)








