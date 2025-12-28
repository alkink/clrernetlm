# V8: LLAMAS Strategy Implementation

## Değişiklikler

### Training Stratejisi Değişti
- **Önceki (V7)**: Lq ◦ Lgt formatı (pseudo label + ground truth)
  - Training'de GT'den Lq (ilk 2 keypoint + noise)
  - Test'te CLRNet'ten Lq (ilk 2 keypoint)
  - **Sorun**: Training/Test uyumsuzluğu, "sudden jump" problemi

- **Yeni (V8)**: LLAMAS Strategy (PDF Section 4.5, Page 12)
  - Training'de: Direkt Lgt (full GT lane) - **Lq yok**
  - Test'te: CLRNet prompting (Lq from CLRNet) - zaten var
  - **Avantaj**: Training/Test uyumsuzluğu yok, "sudden jump" yok

### PDF Referansı
> "Performance on LLAMAS. The result on the LLAMAS is shown in Table 4. LaneLM-512 outperforms PolyLaneNet [20] by 7.05 and LaneATT [8] by 2.08. The training strategy is slightly different with CULane and TuSimple. **We directly use Lgt as self-supervised label S and Lq is not used during training, which is different with Eq. 10. Thus, we can avoid knowledge distilling from the teacher model.**"

### Kod Değişiklikleri

1. **Pseudo Label Kodu Kaldırıldı**
   - Lq ◦ Lgt concatenation kaldırıldı
   - Noise ekleme kaldırıldı
   - Direkt `tokenizer.encode_single_lane(pts)` kullanılıyor

2. **Loss Masking Kaldırıldı**
   - `loss_mask` kaldırıldı
   - Tüm valid token'lar için loss hesaplanıyor
   - X-loss, Y-loss, AR-loss hepsi full sequence için

3. **Teacher Forcing Değişmedi**
   - Normal teacher forcing devam ediyor
   - `x_in_tf[:, 0] = x_tokens[:, 0]` (ilk token GT'den)

### Beklenen Etkiler

1. **Training/Test Uyumsuzluğu Çözüldü**
   - Training'de Lq yok → Test'te CLRNet Lq kullanılabilir (prompting)
   - Model CLRNet keypoint'lerini görmeyi öğrenmemiş ama bu sorun değil
   - Test'te CLRNet prompting sadece "hint" olarak kullanılıyor

2. **"Sudden Jump" Problemi Çözüldü**
   - Lq→Lgt geçişi yok → "sudden jump" yok
   - Model "abrupt change points" pattern'ini öğrenmeyecek

3. **F1@0.5 Skoru İyileşmeli**
   - Önceki: 0.0000 (hiçbir lane IoU@0.5 geçemiyor)
   - Beklenen: >0.1 (minimum), ideal: >0.5

### Test Stratejisi

Test'te CLRNet prompting zaten var (`LaneLMDetector.predict`):
- CLRNet'ten ilk 2 keypoint alınıyor
- `autoregressive_decode`'a `initial_x_tokens` ve `initial_y_tokens` olarak veriliyor
- Model bu keypoint'leri "hint" olarak kullanıyor

### Sonraki Adımlar

1. **Model'i Yeniden Train Et**
   - `python tools/train_lanelm_v4_fixed.py --overfit-size 1` (1-image overfit test)
   - Loss'un düşüp düşmediğini kontrol et

2. **Test Et**
   - Model'i test et ve F1@0.5 skorunu kontrol et
   - Görselleştirmeleri incele (zigzagging azaldı mı?)

3. **İleride: Gerçek CLRNet Pseudo Label (Opsiyonel)**
   - Eğer LLAMAS strategy yeterli olmazsa
   - Training'e CLRNet inference ekle (her batch için)
   - Bipartite matching ekle (PDF Equation 10)
   - Bu training'i yavaşlatır ama daha doğru olur

## Notlar

- LLAMAS strategy basit ve hızlı
- PDF'de LLAMAS için kullanılıyor (CULane/TuSimple'dan farklı)
- Training/test uyumsuzluğunu çözer
- "Sudden jump" problemini ortadan kaldırır








