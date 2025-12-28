# V7: Zigzagging Kalıcı Çözüm Planı

## Kök Neden Analizi

### Test Sonuçları
- **F1@0.5 = 0.0264** (çok kötü)
- **FP@0.5 = 392** (çok fazla)
- **Zigzagging devam ediyor** (görsellerde kırmızı/sarı çizgiler)

### Zigzagging'in Kök Nedenleri

1. **Autoregressive Decode'da Exposure Bias**
   - Model kendi hatalarını biriktiriyor
   - Training'de GT görüyor, inference'da kendi tahminlerini görüyor
   - Küçük hatalar birikerek zigzagging oluşturuyor

2. **Model Visual Bilgiyi Yeterince Kullanmıyor**
   - Geçmiş X token'lara çok bağımlı
   - Cross-attention zayıf (uniform attention)
   - Visual bilgi ikincil sinyal

3. **Tokenization Granularity Çok Kaba**
   - `nbins_x=200` → 800px / 200 = 4px per bin
   - Küçük hatalar birikiyor
   - Smoothness kaybı

4. **Başlangıç Token Problemi**
   - Inference'da padding (0) ile başlıyor
   - Training'de GT ile başlıyor
   - Bu mismatch zigzagging'e neden oluyor

## PDF'den Öğrenilen Kalıcı Çözümler

### 1. Prompting Strategy (PDF Section 3.4)
**"A regression network is employed to provide the two initial keypoints, for each lane. LaneLM is responsible for completing the remaining keypoints."**

**Neden Kalıcı:**
- Başlangıç token problemi çözülüyor
- CLRNet'ten güvenilir başlangıç noktaları alınıyor
- Model sadece devam ediyor, başlangıç hatası yok

**Uygulama:**
- CLRNet'ten ilk 2 keypoint al
- Bu keypoint'leri tokenize et
- Autoregressive decode'da bu token'ları kullan

### 2. Hallucination Removal (HR) (PDF Section 3.5, Algorithm 1)
**"Points with offsets of adjacent x-coordinates exceeding twice the 85th percentile, along with their subsequent points, will be filtered out."**

**Neden Kalıcı:**
- Zigzagging'i post-processing'de filtreliyor
- Anormal sıçramaları tespit ediyor
- PDF'de kanıtlanmış çözüm

**Uygulama:**
```python
def hallucination_removal(x_coords, N=10):
    if len(x_coords) > N:
        diff = np.abs(np.diff(x_coords))
        theta = 2 * np.percentile(diff, 85)
        p = np.argmax(diff > theta)
        if p > 0:
            x_coords = x_coords[:p+1]
    return x_coords
```

### 3. Tokenization Granularity Artırma
**PDF'de `nbins_x=800` kullanılıyor (bizde 200)**

**Neden Kalıcı:**
- Daha ince granularity → daha smooth predictions
- Küçük hatalar birikmiyor
- Model daha hassas öğrenebilir

**Uygulama:**
- `nbins_x: 200 → 400` (veya 800)
- Model'i yeniden eğitmek gerekiyor

### 4. Smoothness Loss (Geometric)
**Model'in smoothness öğrenmesini sağlamak**

**Neden Kalıcı:**
- Model seviyesinde smoothness öğreniliyor
- Post-processing'e bağımlı değil
- Training'de smoothness zorunlu

**Uygulama:**
- Second derivative loss (curvature)
- Adjacent token difference loss
- Training'de smoothness zorunlu

## Uygulama Önceliği

### Faz 1: Hallucination Removal (HR) - Hemen
- Post-processing çözümü
- Model eğitimi gerektirmiyor
- PDF'de kanıtlanmış

### Faz 2: Prompting Strategy - Orta Vadeli
- CLRNet entegrasyonu
- Autoregressive decode güncellemesi
- Model eğitimi gerektirmiyor (sadece inference)

### Faz 3: Tokenization Granularity - Uzun Vadeli
- Model architecture değişikliği
- Model'i yeniden eğitmek gerekiyor
- En kalıcı çözüm

### Faz 4: Smoothness Loss - Uzun Vadeli
- Training strategy değişikliği
- Model'i yeniden eğitmek gerekiyor
- En kalıcı çözüm

## Beklenen Etki

### Önceki Durum
- F1@0.5 = 0.0264
- FP@0.5 = 392
- Zigzagging: Yüksek

### Faz 1 Sonrası (HR)
- F1@0.5 = 0.1-0.2 (FP azalması)
- Zigzagging: Orta (anormal sıçramalar filtreleniyor)

### Faz 2 Sonrası (Prompting)
- F1@0.5 = 0.2-0.3 (başlangıç hatası yok)
- Zigzagging: Düşük (başlangıç doğru)

### Faz 3+4 Sonrası (Granularity + Smoothness)
- F1@0.5 = 0.4-0.6 (tam çözüm)
- Zigzagging: Çok düşük (model seviyesinde smooth)

## Sonraki Adımlar

1. ✅ HR algoritmasını implement et
2. ⏳ Test script'ine HR ekle
3. ⏳ Prompting strategy implement et
4. ⏳ Tokenization granularity artır
5. ⏳ Smoothness loss ekle








