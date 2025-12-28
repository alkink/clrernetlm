## PRD: V24 — 0‑KP Overfit Ezber Var ama Inference Kötü (AR BOS Mismatch)

### Problem
0‑kp (`--num-pseudo-points 0`) overfit koşusunda eğitim:
- `Loss(X) ≈ 0.0019`
- `ACC = 1.0`

olmasına rağmen inference görselleri hala “zigzag / yukarı fırlayan şeritler” üretiyor.

Bu durum tipik olarak **teacher-forcing ile öğrenilen dağılım** ile **autoregressive (free-run) decode girişlerinin farklı** olmasına işaret eder.

### Kök Neden
`libs/models/detectors/lanelm_detector.py` içindeki `autoregressive_decode` fonksiyonunda:
- `t>0` adımlarında `x_in[:,0] = x_out[:,0]` yapılıyordu.
- Ancak 0‑kp eğitimde (`train_lanelm_v4_fixed.py`) teacher forcing girişinde:
  - `x_in_tf[:,0] = pad_token_x (0)` **her zaman**.

Yani inference, her adımda ilk pozisyona “tahmin edilen x0” koyarak modele **eğitimde hiç görmediği** bir giriş dağılımı veriyordu. Bu da özellikle ufuk bölgesinde hızla sapma (zigzag / upward rise) üretir.

### Çözüm
0‑kp modunda inference BOS/pad hizası eğitimle eşitlendi:
- Prompting yoksa: `x_in[:,0] = pad_token_x`
- Prompting varsa (ilk 2 keypoint verilmişse): `x_in[:,0] = prompt x0` korunur.

### Değişiklik
- Dosya: `libs/models/detectors/lanelm_detector.py`
- Bölüm: `autoregressive_decode`, `t>0` branch
- Dosya: `tools/train_lanelm_v4_fixed.py`
- Bölüm: `visual_first_decode` (training-side görselleştirme / debug decode)

### Kabul Kriterleri
- `work_dirs/v22_overfit1_0kp/lanelm_v4_best.pth` ile **aynı görüntü** üzerinde inference çıktısı:
  - Training görseline yakın şekilde GT’ye oturmalı
  - Zigzag/upward artefact belirgin şekilde azalmalı


