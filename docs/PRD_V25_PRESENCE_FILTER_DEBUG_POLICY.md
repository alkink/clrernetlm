## PRD: V25 — Presence Filter Debug Politikası (Overfit/0‑KP)

### Problem
Overfit/0‑kp debug koşularında eğitim loglarında `PRES≈0.7` görülse bile,
çoğu senaryoda `presence_weight=0` olduğu için bu loss **toplam loss’a eklenmiyor**.
Dolayısıyla presence head çıktıları pratikte **rastgele** kalıyor.

Inference tarafında `use_presence_filter=True` olduğunda:
- lane’ler presence skoruna göre eleniyor,
- kalan lane’ler “reorder” olup `lane0` artık gerçek `lane_idx=0` olmayabiliyor,
- görseller/karşılaştırmalar ciddi şekilde bozuluyor.

Bu durum kullanıcıda “model ezberledi ama inference berbat” algısı yaratıyor.

### Çözüm
- Overfit/0‑kp debug checkpoint’i ile test ederken **presence_filter kapalı** olmalı.
- Presence filter kullanılacaksa:
  - `presence_weight > 0` ile presence head **eğitilmeli**,
  - sonra `use_presence_filter=True` yeniden açılmalı.

### Yapılan Değişiklikler
- `configs/lanelm/lanelm_v4_culane_test.py`: `use_presence_filter=False` (debug ckpt için)
- `tools/debug_training_vs_test_tokens.py`: test path `use_presence_filter=False` (lane reorder engeli)

### Kabul Kriterleri
- `debug_training_vs_test_tokens.py` çıktısında:
  - token/coord farkları presence filtre yüzünden “suni” büyümemeli
  - lane0 karşılaştırması gerçekten lane_idx=0 ile yapılmalı



