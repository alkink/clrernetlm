# V12: Çözüm - "Abrupt Change Points" Problemi

## Sorun Özeti

**Test Sonuçları:**
- F1@0.5: 0.0000-0.0626 (çok kötü)
- Kullanıcı: "yolları da çok iyi coverlamıyor" - Geometrik olarak da yanlış tahmin ediyor

**PDF'den Kritik Bulgu:**
- PDF sayfa 879-885: "a sudden jump occurs at the junction between the pseudo-label and the ground truth (see Eq. 10), which disrupts the contextual semantic information and confuses the model."
- PDF sayfa 1619-1623: "the model has learned the abrupt change points that connecting Lq and Lgt on the side. LaneLM has learned the contextual representation of abrupt change points and consequently results in hallucination."

**Root Cause:** Model Lq ve Lgt arasındaki "abrupt change points" öğreniyor ve bu hallucination'a yol açıyor!

## Çözüm: Training'i Değiştir

### 1. Noise'u Artır (PDF Section 3.4)

PDF'de noise -5 to +5 pixels. Bunu artıralım:
- Şu an: `lq_noise_range = 5`
- Öneri: `lq_noise_range = 10` (PDF'de belirtilen maksimum)

### 2. Lq Loss Ekle (Düşük Weight ile)

PDF'de loss sadece Lgt kısmında hesaplanıyor. Belki Lq kısmında da loss hesaplayabiliriz ama düşük weight ile:
- Lq loss weight: 0.1
- Lgt loss weight: 1.0

Bu model'e Lq'nun da önemli olduğunu öğretir ve "abrupt change" problemini azaltır.

### 3. Test'te x_in'i Doğru Oluştur

Test'te x_in'i padding ile doldurmalıyız çünkü model forward'da tüm sequence'i paralel işliyor.

## Uygulama

1. `lq_noise_range`'i 5'ten 10'a çıkar
2. Lq loss ekle (weight 0.1)
3. Test'te x_in'i padding ile doldur (zaten yapıldı)

## Notlar

- PDF'de "abrupt change points" problemi açıkça belirtilmiş
- Model Lq ve Lgt arasındaki "abrupt change" öğreniyor
- Training'de loss sadece Lgt kısmında hesaplanıyor
- Çözüm: Noise'u artır, Lq loss ekle






