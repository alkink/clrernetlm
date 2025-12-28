# V12: CULane "*" Version (PDF'den)

## Değişiklikler

### Training Stratejisi Değişti
- **Önceki (V11)**: LLAMAS Strategy - No Lq in Training
  - Lq kullanmıyorduk (ne CLRNet'ten, ne GT'den)
  - Direkt GT keypoint'leri alıyorduk (Lgt)
  - **Sorun**: Model CLRNet keypoint'lerini yorumlamayı öğrenmemiş (test'te CLRNet Lq var)

- **Yeni (V12)**: PDF "*" Version for CULane
  - CLRNet'ten 2 keypoint alıyoruz (Lq) - bipartite matching ile
  - GT'den kalan keypoint'leri alıyoruz (Lgt)
  - **Avantaj**: Training/test uyumlu (Her ikisinde de CLRNet Lq)

### PDF Referansı (Sayfa 867-871)

> "Our model receives two adjacent keypoints output from CLRNet [6] as init prompts for each lane and rollouts the remaining keypoints in the * version."

**KRİTİK:** PDF'de CULane için "*" versiyonu kullanılıyor, LLAMAS strategy değil!

**LLAMAS Strategy (Sayfa 887-891):**
> "The training strategy is slightly different with CULane and TuSimple. We directly use Lgt as self-supervised label S and Lq is not used during training..."

**LLAMAS strategy sadece LLAMAS dataset'i için, CULane için değil!**

### Kod Değişiklikleri

1. **CLRNet Inference Eklendi**
   - Training'de CLRNet inference var
   - Bipartite matching var
   - CLRNet'ten 2 keypoint alınıyor (Lq)

2. **Lq ◦ Lgt Formatı**
   - Lq = CLRNet'ten ilk 2 keypoint (random shift ile)
   - Lgt = GT'den kalan keypoint'ler
   - Concatenate: Lq ◦ Lgt

3. **Loss Masking**
   - Loss sadece Lgt kısmında hesaplanıyor (Lq kısmı input, loss yok)
   - X-loss, Y-loss, AR-loss hepsi Lgt için

### Beklenen Etkiler

1. **Training/Test Uyumlu**
   - Training: CLRNet Lq ◦ GT Lgt
   - Test: CLRNet Lq (prompting)
   - **Training/test uyumlu!** (Her ikisinde de CLRNet Lq)

2. **Model CLRNet Keypoint'lerini Yorumlamayı Öğrenir**
   - Training'de CLRNet Lq görüyor
   - Test'te CLRNet Lq kullanıyor
   - Model CLRNet keypoint'lerini yorumlamayı öğrenir

3. **F1@0.5 Skoru İyileşmeli**
   - Önceki: 0.0132 (çok kötü)
   - Beklenen: >0.1 (minimum), ideal: >0.3
   - PDF'de "*" versiyonu F1@0.5 = 79.04 (CULane)

### Dezavantajlar

1. **CLRNet'in Hatalarını Öğrenir (Knowledge Distillation)**
   - Training'de CLRNet Lq kullanılıyor
   - Model CLRNet'in hatalarını öğrenir
   - **Ama PDF'de bu kabul edilebilir (F1@0.5 = 79.04)**

2. **"Sudden Jump" Problemi**
   - Lq ve Lgt arasındaki geçişte "sudden jump" var
   - Noise eklemek yeterli olmayabilir
   - **Ama PDF'de bu çalışıyor**

### Sonraki Adımlar

1. **Model'i Yeniden Train Et**
   - `python tools/train_lanelm_v4_fixed.py --overfit-size 1 --epochs 200`
   - Loss'un düşüp düşmediğini kontrol et

2. **Test Et**
   - Model'i test et ve F1@0.5 skorunu kontrol et
   - Görselleştirmeleri incele (zigzagging azaldı mı?)

3. **Analiz Et**
   - Training/test uyumlu mu?
   - Model CLRNet keypoint'lerini yorumlamayı öğrendi mi?
   - F1@0.5 skoru iyileşti mi?

## Notlar

- PDF'nin CULane "*" versiyonuna uygun
- Training/test uyumlu
- Model CLRNet keypoint'lerini yorumlamayı öğrenir
- PDF'de CULane için çalışıyor (F1@0.5 = 79.04)
- LLAMAS strategy CULane için değil, LLAMAS dataset'i için






