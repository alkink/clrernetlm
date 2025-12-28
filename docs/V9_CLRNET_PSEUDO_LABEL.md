# V9: CLRNet Pseudo Label Strategy (PDF Equation 10)

## Değişiklikler

### Training Stratejisi Değişti
- **Önceki (V8 - LLAMAS)**: Direkt Lgt (full GT lane) - Lq yok
  - Training'de Lq yok → Test'te CLRNet Lq kullanılamıyor
  - **Sorun**: Training/Test uyumsuzluğu, model CLRNet keypoint'lerini yorumlayamıyor

- **Yeni (V9)**: CLRNet Lq ◦ GT Lgt (PDF Equation 10)
  - Training'de: CLRNet'ten Lq (ilk 2 keypoint + noise) + GT'den Lgt (kalan keypoint'ler)
  - Test'te: CLRNet prompting (Lq from CLRNet) - zaten var
  - **Avantaj**: Training ve test aynı kaynağı kullanır (CLRNet), model CLRNet keypoint'lerini öğrenir

### PDF Referansı
> "We adopt the bipartite matching to find the matching that minimizes the distance of the start points between the query sequence Li_q and the answer Lj_gt" (PDF Equation 10)

> "randomly shifting the x-coordinates by -5 to 5 pixels" (PDF Section 3.4)

### Kod Değişiklikleri

1. **Full CLRNet Model Build**
   - `init_detector` ile full CLRNet model (head dahil) build edildi
   - Eval mode'da, tüm parametreler frozen

2. **CLRNet Inference Her Batch İçin**
   - Her batch için `clrernet_full.predict()` çağrılıyor
   - CLRNet'ten lane predictions alınıyor

3. **Bipartite Matching (PDF Eq. 10)**
   - Start point distance kullanılarak CLRNet lanes ↔ GT lanes eşleştiriliyor
   - `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) kullanılıyor
   - Cost matrix: start point Euclidean distance

4. **Lq ◦ Lgt Formatı**
   - Lq = CLRNet'ten ilk 2 keypoint (noise ile)
   - Lgt = GT'den kalan keypoint'ler
   - Concatenate: Lq ◦ Lgt

5. **Noise Simulation (PDF Section 3.4)**
   - CLRNet Lq keypoint'lerine random noise (-5 to +5 pixels) ekleniyor
   - Bu, "sudden jump" problemini azaltır

6. **Loss Masking**
   - Loss sadece Lgt kısmında hesaplanıyor (Lq kısmı input, loss yok)
   - X-loss, Y-loss, AR-loss hepsi Lgt için

### Beklenen Etkiler

1. **Training/Test Uyumsuzluğu Çözüldü**
   - Training'de CLRNet Lq → Test'te CLRNet Lq
   - Model CLRNet keypoint'lerini öğrenir

2. **"Sudden Jump" Problemi Azaldı**
   - Noise ekleme ile Lq ve Lgt arasındaki süreksizlik azalır
   - Model daha smooth geçişler öğrenir

3. **F1@0.5 Skoru İyileşmeli**
   - Önceki: 0.03 (çok kötü)
   - Beklenen: >0.1 (minimum), ideal: >0.5

### Dezavantajlar

1. **Training Yavaşlar**
   - Her batch için CLRNet inference gerekir
   - Bipartite matching hesaplaması eklenir

2. **"Abrupt Change Points" Öğrenme Riski**
   - PDF'de belirtildiği gibi, model Lq→Lgt geçişindeki "abrupt change" pattern'ini öğrenebilir
   - Noise ekleme bu riski azaltır ama tamamen ortadan kaldırmaz

### Sonraki Adımlar

1. **Model'i Yeniden Train Et**
   - `python tools/train_lanelm_v4_fixed.py --overfit-size 1 --epochs 200`
   - Loss'un düşüp düşmediğini kontrol et

2. **Test Et**
   - Model'i test et ve F1@0.5 skorunu kontrol et
   - Görselleştirmeleri incele (zigzagging azaldı mı?)

3. **Analiz Et**
   - Training/test uyumsuzluğu çözüldü mü?
   - F1@0.5 skoru iyileşti mi?
   - Zigzagging azaldı mı?

## Notlar

- PDF'nin önerdiği strateji (Eq. 10)
- Training/test uyumsuzluğunu çözer
- Model CLRNet keypoint'lerini öğrenir
- Bipartite matching ile doğru eşleştirme sağlanır








