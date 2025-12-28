## PRD: V26 — Train_100 → Test_100 “Gerçek Eğitim” Döngüsü (0‑KP, Full FPN, 512)

### Amaç
Overfit (1 img) debug checkpoint’i yerine, `train_100` üzerinde gerçek bir eğitim yapıp `test_100` üzerinde anlamlı görsel kalite ve metrik değerlendirmesi almak.

### Neden Gerekli?
`v22_overfit1_0kp/lanelm_v4_best.pth` sadece 1 görüntüyü ezberlemek için üretildi.
Bu checkpoint ile `test_100` görsellerinin kötü olması beklenen bir durum (genelleme yok).

### Önkoşullar / Düzeltmeler
- V24: 0‑kp BOS/pad mismatch düzeltildi (AR decode giriş dağılımı eğitimle hizalı).
- V25: Presence filter, presence head eğitilmediyse kapalı tutulmalı (lane reorder / görsel bozulma).

### Eğitim Konfigürasyonu (öneri)
- `tools/train_lanelm_v4_fixed.py`
- `list-path`: `dataset/list/train_100.txt`
- `overfit-size`: 100 (bu scriptte batch_size=8’e geçmek için; dataset zaten 100 ise subset olmaz)
- `num-pseudo-points`: 0 (0‑kp)
- Loss: X-only (başlangıç)
- Presence: kapalı (presence_weight=0) — testte presence_filter kapalı

### Komutlar
#### 1) Train (train_100)
```bash
cd /home/alki/projects/clrernetlm && /home/alki/miniconda3/envs/clrernet/bin/python tools/train_lanelm_v4_fixed.py \
  --list-path dataset/list/train_100.txt \
  --overfit-size 100 \
  --epochs 50 \
  --num-pseudo-points 0 \
  --x-embedding-scale 1.0 \
  --lane-embedding-boost 5.0 \
  --ss-max-prob 0.2 \
  --ar-rollout-max-weight 0.1 \
  --ar-rollout-min-weight 0.05 \
  --presence-weight 0.0 \
  --work-dir work_dirs/v26_train100_0kp \
  --device cuda
```

#### 2) Test (test_100)
```bash
cd /home/alki/projects/clrernetlm && /home/alki/miniconda3/envs/clrernet/bin/python tools/test_lanelm_runner.py \
  configs/lanelm/lanelm_v4_culane_test_v26_train100.py \
  --work-dir work_dirs/v26_test100
```

#### 3) Görselleştirme (prediction dosyalarından)
```bash
cd /home/alki/projects/clrernetlm && /home/alki/miniconda3/envs/clrernet/bin/python tools/visualize_culane_predictions_from_files.py \
  --pred-root work_dirs/v26_test100/predictions \
  --data-root dataset \
  --data-list dataset/list/test_100.txt \
  --out-dir work_dirs/v26_test100/visualizations \
  --max-samples 50
```

### Kabul Kriterleri
- `work_dirs/v26_train100_0kp/lanelm_v4_best.pth` oluşmalı.
- `test_100` görsellerinde:
  - “yukarı fırlama / aşırı zigzag” artefact’ları belirgin azalmalı.
  - Lane’ler GT’ye daha iyi oturmalı.



