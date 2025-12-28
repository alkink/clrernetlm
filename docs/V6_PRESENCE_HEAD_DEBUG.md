# V6 Presence Head Debug Analizi

## Problem: TP=0, FP=0, FN=207

**Durum:** Test sonuçları hiç lane tahmin edilmediğini gösteriyor (TP=0, FP=0, FN=207).

## Olası Nedenler

### 1. Presence Head Filtrelemesi Çok Agresif
- Presence threshold=0.5 çok yüksek olabilir
- Presence logits'ler düşük olabilir (model presence head'i öğrenmemiş olabilir)
- Presence head training'de yeterince optimize edilmemiş olabilir

### 2. Presence Head Hesaplama Hatası
- `return_presence=True` ile presence logits hesaplanıyor ama değerler yanlış olabilir
- Hidden state pooling'de sorun olabilir
- Presence head weights yüklenmemiş olabilir

### 3. PDF'deki LaneLM ile Fark
- **PDF'de presence head YOK** - sadece EOS token (x=0 veya y=T) kullanıyorlar
- Inference'da EOS token ile durduruyorlar
- Hallucination Removal (HR) kullanıyorlar ama presence head değil

## Debug Adımları

### 1. Presence Logits Analizi
```bash
python tools/debug_presence_head.py --num-samples 10
```

Bu script:
- Presence logits'leri loglar
- Presence probabilities'leri gösterir
- Farklı threshold'ları test eder
- Öneriler sunar

### 2. Presence Filtering'i Geçici Olarak Kapat
`lanelm_detector.py` içinde:
```python
use_presence_filter=False,  # DEBUG: Disable presence filtering
```

Bu, presence head'in sorun olup olmadığını anlamamıza yardımcı olur.

### 3. Threshold Varyasyonu
Farklı threshold'ları test et:
- 0.0 (tüm lane'leri geçir)
- 0.1 (çok düşük)
- 0.3 (orta)
- 0.5 (mevcut)
- 0.7 (yüksek)

### 4. Prediction Dosyalarını Kontrol Et
```bash
ls -la work_dirs/lanelm_v4_test_fixed_100/predictions/
head work_dirs/lanelm_v4_test_fixed_100/predictions/*.lines.txt | head -20
```

Eğer dosyalar boşsa veya hiç lane yoksa, presence filtering çok agresif demektir.

## Çözüm Önerileri

### Kısa Vadeli (Hemen)
1. **Presence filtering'i kapat** - EOS token ile durdurma kullan (PDF'deki gibi)
2. **Threshold'u düşür** - 0.5 → 0.1-0.3
3. **Fallback mekanizması** - Hiç lane geçmezse, en yüksek presence score'a sahip lane'i kullan

### Orta Vadeli
1. **Presence head'i yeniden eğit** - Daha agresif loss weight (0.3 → 0.5-1.0)
2. **Presence head architecture'ını güçlendir** - Daha derin network
3. **Training'de presence loss'u artır** - Model presence head'i daha iyi öğrensin

### Uzun Vadeli
1. **PDF'deki stratejiyi uygula** - Presence head yerine EOS token kullan
2. **Hallucination Removal (HR) ekle** - PDF'deki Algorithm 1'i implement et
3. **Prompting strategy ekle** - CLRNet'ten initial keypoints al (PDF Section 3.4)

## Sonraki Adımlar

1. ✅ Debug script oluşturuldu (`tools/debug_presence_head.py`)
2. ⏳ Debug script'i çalıştır ve presence logits'leri analiz et
3. ⏳ Presence filtering'i geçici olarak kapat ve test et
4. ⏳ Threshold varyasyonu test et
5. ⏳ PDF'deki EOS token stratejisini implement et (presence head yerine)








