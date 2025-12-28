## PRD: V23 — 0‑KP Overfit Plato (X-loss ~0.98) Kök Neden İzolasyonu

### Amaç
`--num-pseudo-points 0` (0‑kp) overfit koşusunda 200 epoch sonunda X-loss’un ~0.98’de plato yapmasının kök nedenini izole etmek ve modelin **gerçek ezberi** (loss → ~0) yapmasını sağlamak.

### Gözlem
- 0‑kp, X-only koşusunda loss 6.8 → 0.98 düşüyor ama 0’a yaklaşmıyor.
- Bu, modelin “kısmi” öğrenip **görüntüye göre tam ayrışamadığını** gösterir.

### Hipotezler (yüksek olasılık)
1) **Lane slot embedding baskınlığı**
   - `KeypointEmbedding` içinde `lane_embedding_boost` yüksekse (15 gibi), model lane_id sinyaline aşırı yaslanıp görselden ayrışmayı zayıflatabilir.
2) **AR geçmiş X sinyalinin zayıflatılması**
   - `x_embedding_scale=0.3` gibi düşük değerler AR sinyalini azaltır; overfit debug’da ezberi zorlaştırabilir.
3) **GT lane slot permütasyonu**
   - Dataset içinde lane sırası tutarlı değilse `lane_slot` etiketi görüntüler arasında farklı anlamlara gelir; bu da ezberi zorlaştırır.

### Yapılan Değişiklikler
#### 1) `LaneLMModel` içinde embedding ölçeklerini parametreleştirme
- Dosya: `libs/models/lanelm/model.py`
- `LaneLMModel(..., x_embedding_scale=..., lane_embedding_boost=...)` eklendi.
- Böylece overfit debug’da AR sinyali güçlendirilip lane_id baskınlığı azaltılabilir.

#### 2) Overfit scriptine yeni debug flag’ler
- Dosya: `tools/train_lanelm_v4_fixed.py`
- Yeni argümanlar:
  - `--x-embedding-scale` (default 1.0)
  - `--lane-embedding-boost` (default 5.0)
  - `--no-sort-gt-lanes` (default: sıralama açık)

#### 3) GT lane’leri soldan‑sağa sıralama
- Varsayılan açık; `--no-sort-gt-lanes` ile kapatılabilir.
- Amaç: lane_slot etiketini görüntüler arasında daha tutarlı yapmak.

#### 4) Debug metrik: token accuracy
- Log’a `ACC=...` eklendi (loss mask üzerindeki valid target pozisyonlarında).

### Kabul Kriterleri
- Overfit size 8, 0‑kp, X-only koşusunda:
  - `ACC` belirgin artmalı (ideal: 0.95+)
  - `Loss(X)` belirgin düşmeli (ideal: <0.1, hedef: ~0)

### Önerilen Komut
```bash
/home/alki/miniconda3/envs/clrernet/bin/python tools/train_lanelm_v4_fixed.py \
  --overfit-size 8 --epochs 200 --num-pseudo-points 0 \
  --x-embedding-scale 1.0 --lane-embedding-boost 5.0 \
  --work-dir work_dirs/v23_overfit8_0kp \
  --device cuda
```






