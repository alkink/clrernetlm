# V11: LLAMAS Strategy (PDF'den)

## Değişiklikler

### Training Stratejisi Değişti
- **Önceki (V10)**: GT Lq ◦ GT Lgt (PDF "(2-kp)" strategy)
  - GT'den ilk 2 keypoint alıyorduk (Lq) - random shift ile
  - GT'den kalan keypoint'leri alıyorduk (Lgt)
  - **Sorun**: Training/test uyumsuzluğu (Training: GT Lq, Test: CLRNet Lq)

- **Yeni (V11)**: Direkt GT Lgt (LLAMAS strategy)
  - Lq kullanmıyoruz (ne CLRNet'ten, ne GT'den)
  - Direkt GT keypoint'lerini alıyoruz (Lgt)
  - **Avantaj**: Training/test uyumsuzluğu yok (Her ikisinde de Lq yok training'de)

### PDF Referansı (Sayfa 887-891)

> "The training strategy is slightly different with CULane and TuSimple. We directly use Lgt as self-supervised label S and **Lq is not used during training**, which is different with Eq. 10. Thus, we can avoid knowledge distilling from the teacher model."

**KRİTİK:** PDF'de LLAMAS strategy training'de Lq kullanmıyor!

### Kod Değişiklikleri

1. **Lq Kaldırıldı**
   - Training'de Lq yok (ne CLRNet'ten, ne GT'den)
   - Direkt GT keypoint'leri alınıyor (Lgt)
   - Lq ◦ Lgt formatı yok, sadece Lgt

2. **Loss Masking Kaldırıldı**
   - Loss mask artık tüm valid token'lar için True
   - Lq masking yok (çünkü Lq yok)

3. **X-Loss, Y-Loss, AR-Loss**
   - Tüm loss'lar tüm valid token'lar için hesaplanıyor
   - Lq masking yok

### Beklenen Etkiler

1. **Training/Test Uyumsuzluğu Çözüldü**
   - Training: Lq yok, direkt Lgt
   - Test: CLRNet Lq (prompting) - Model bunları yorumlamayı öğrenmemiş ama PDF'de çalışıyor

2. **CLRNet'in Hatalarını Öğrenme Problemi Çözüldü**
   - Training'de Lq kullanılmıyor → Model CLRNet'in hatalarını öğrenmez

3. **"Sudden Jump" Problemi Çözüldü**
   - Lq yok → "Sudden jump" problemi yok

4. **F1@0.5 Skoru İyileşmeli**
   - Önceki: 0.0165 (çok kötü)
   - Beklenen: >0.1 (minimum), ideal: >0.3
   - PDF'de LLAMAS'da F1: 97.25 (TuSimple), CULane için benzer iyileşme bekleniyor

### Dezavantajlar

1. **Model CLRNet Keypoint'lerini Yorumlamayı Öğrenmez**
   - Training'de Lq yok
   - Test'te CLRNet Lq kullanılıyor
   - Model CLRNet keypoint'lerini yorumlamayı öğrenmemiş
   - **Ama PDF'de LLAMAS'da bu çalışıyor!**

### Sonraki Adımlar

1. **Model'i Yeniden Train Et**
   - `python tools/train_lanelm_v4_fixed.py --overfit-size 1 --epochs 200`
   - Loss'un düşüp düşmediğini kontrol et

2. **Test Et**
   - Model'i test et ve F1@0.5 skorunu kontrol et
   - Görselleştirmeleri incele (zigzagging azaldı mı?)

3. **Analiz Et**
   - Training/test uyumsuzluğu problemi çözüldü mü?
   - F1@0.5 skoru iyileşti mi?
   - Zigzagging azaldı mı?

## Notlar

- PDF'nin LLAMAS strategy'sine uygun
- Training/test uyumsuzluğu yok
- CLRNet'in hatalarını öğrenmeyi önler
- "Sudden jump" problemi yok
- PDF'de LLAMAS'da çalışıyor (F1: 97.25)






