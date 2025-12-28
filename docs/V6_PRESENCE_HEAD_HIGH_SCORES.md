# V6: Presence Head Her Zaman Yüksek Skor Veriyor

## Kritik Bulgu

Presence head çalışıyor ama **TÜM LANE'LER threshold'u geçiyor**:

```
[DEBUG] Lane 0: presence_prob=0.9911, presence_logit=4.7077
[DEBUG] Lane 1: presence_prob=0.9926, presence_logit=4.8988
[DEBUG] Lane 2: presence_prob=0.9775, presence_logit=3.7719
[DEBUG] Lane 3: presence_prob=0.9970, presence_logit=5.7953
[DEBUG] Presence filtering: threshold=0.3
[DEBUG]   Lane 0: score=0.9911, pass=True
[DEBUG]   Lane 1: score=0.9926, pass=True
[DEBUG]   Lane 2: score=0.9775, pass=True
[DEBUG]   Lane 3: score=0.9970, pass=True
[DEBUG]   Total lanes passing filter: 4/4
```

**Sorun:** Presence head **her zaman yüksek skor veriyor** (0.97-0.99). Bu, presence head'in öğrenmediği veya yanlış öğrendiği anlamına geliyor.

## Olası Nedenler

### 1. Training'de Her Zaman Pozitif Örnekler
- Training'de her zaman 4 lane slot'u dolu olabilir
- Presence head sadece pozitif örneklerle eğitilmiş olabilir
- Negatif örnekler (padding lanes) yeterince eğitilmemiş olabilir

### 2. Presence Loss Weight Çok Düşük
- Presence loss weight çok düşük olabilir (0.3)
- Token loss presence loss'u domine ediyor olabilir
- Presence head öğrenemiyor olabilir

### 3. Pooling Stratejisi Yanlış
- Presence head'in pooling stratejisi yanlış olabilir
- Valid token mask'i doğru çalışmıyor olabilir
- Pooled hidden state yanlış hesaplanıyor olabilir

## Geçici Çözüm

### Threshold'u Yükseltmek
- Threshold'u 0.95'e yükselttik
- Bu, sadece en yüksek skorlu lane'leri geçirecek
- Ama bu geçici bir çözüm, asıl sorun presence head'in öğrenmemesi

## Kalıcı Çözüm

### 1. Presence Head'i Yeniden Eğitmek
- Training'de presence loss weight'i artırmak (0.3 → 0.5, 0.7)
- Negatif örnekleri daha fazla eğitmek
- Presence head'in öğrenmesini sağlamak

### 2. Alternatif Filtreleme
- Presence head yerine valid token sayısına göre filtreleme
- En yüksek skorlu N lane'i seçme
- Geometric filtreleme (uzunluk, eğrilik, vb.)

## Sonraki Adımlar

1. ✅ Threshold'u 0.95'e yükselttik (geçici çözüm)
2. ⏳ Test script'ini çalıştır ve sonuçları kontrol et
3. ⏳ Eğer hala FP yüksekse, presence head'i yeniden eğitmek gerekebilir
4. ⏳ Alternatif filtreleme stratejileri düşünmek








