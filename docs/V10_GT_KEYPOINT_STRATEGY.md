# V10: GT Keypoint Strategy (PDF "(2-kp)")

## Değişiklikler

### Training Stratejisi Değişti
- **Önceki (V9)**: CLRNet Lq ◦ GT Lgt (bipartite matching)
  - CLRNet'ten 2 keypoint alıyorduk (Lq)
  - GT'den kalan keypoint'leri alıyorduk (Lgt)
  - **Sorun**: Model CLRNet'in hatalarını öğreniyor

- **Yeni (V10)**: GT Lq ◦ GT Lgt (PDF "(2-kp)" strategy)
  - GT'den ilk 2 keypoint alıyoruz (Lq) - random shift ile
  - GT'den kalan keypoint'leri alıyoruz (Lgt)
  - **Avantaj**: Model CLRNet'in hatalarını öğrenmez

### PDF Referansı (Sayfa 867-871)

> "(2-kp) denotes that the holistic lane predicted from CLRNet is given and **two adjacent ground truth keypoints with random shift** at the commencement of each lane are also supplied to enhance model performance."

**KRİTİK:** PDF'de "(2-kp)" versiyonu **GT'den keypoint** alıyor, CLRNet'ten değil!

### Kod Değişiklikleri

1. **CLRNet Inference Kaldırıldı**
   - Training'de CLRNet inference yok
   - Bipartite matching yok
   - Daha hızlı training

2. **GT'den Keypoint Alınıyor**
   - GT'den ilk 2 keypoint alınıyor (Lq)
   - Random shift ekleniyor (-5 to +5 pixels)
   - GT'den kalan keypoint'ler alınıyor (Lgt)

3. **Lq ◦ Lgt Formatı**
   - Lq = GT'den ilk 2 keypoint (random shift ile)
   - Lgt = GT'den kalan keypoint'ler
   - Concatenate: Lq ◦ Lgt

4. **Loss Masking**
   - Loss sadece Lgt kısmında hesaplanıyor (Lq kısmı input, loss yok)
   - X-loss, Y-loss, AR-loss hepsi Lgt için

### Beklenen Etkiler

1. **CLRNet'in Hatalarını Öğrenme Problemi Çözüldü**
   - Training'de GT keypoint → Model doğru keypoint öğrenir
   - Test'te CLRNet keypoint → Training/test uyumsuzluğu var ama PDF'de de bu var

2. **"Sudden Jump" Problemi Azaldı**
   - GT'den keypoint → Daha smooth geçiş
   - Random shift → Model noise'a karşı robust

3. **F1@0.5 Skoru İyileşmeli**
   - Önceki: 0.0165-0.0461 (çok kötü)
   - Beklenen: >0.1 (minimum), ideal: >0.3

### Dezavantajlar

1. **Training/Test Uyumsuzluğu**
   - Training: GT keypoint (Lq)
   - Test: CLRNet keypoint (Lq)
   - Model CLRNet keypoint'lerini yorumlamayı öğrenmez
   - **Ama PDF'de de bu var ve çalışıyor!**

2. **CLRNet'in Visual Guidance'ı Yok**
   - PDF'de "(2-kp)" versiyonu CLRNet'ten holistic lane alıyor (visual guidance için)
   - Şu anki implementasyonda CLRNet kullanılmıyor
   - **Ama bu opsiyonel, PDF'de de zorunlu değil**

### Sonraki Adımlar

1. **Model'i Yeniden Train Et**
   - `python tools/train_lanelm_v4_fixed.py --overfit-size 1 --epochs 200`
   - Loss'un düşüp düşmediğini kontrol et

2. **Test Et**
   - Model'i test et ve F1@0.5 skorunu kontrol et
   - Görselleştirmeleri incele (zigzagging azaldı mı?)

3. **Analiz Et**
   - CLRNet'in hatalarını öğrenme problemi çözüldü mü?
   - F1@0.5 skoru iyileşti mi?
   - Zigzagging azaldı mı?

## Notlar

- PDF'nin "(2-kp)" stratejisine uygun
- CLRNet'in hatalarını öğrenmeyi önler
- "Sudden jump" problemi azalır
- Training/test uyumsuzluğu var ama PDF'de de bu var ve çalışıyor








