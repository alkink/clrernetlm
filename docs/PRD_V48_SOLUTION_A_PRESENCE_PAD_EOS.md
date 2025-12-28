## PRD_V48 — Çözüm A: Presence + Pad/EOS Öğretimi (Lane sayısı + uzunluk)

### Hedef
- **Lane sayısı**: ekstra lane’leri azalt (FP ↓) → `presence head` + `use_presence_filter`
- **Lane uzunluğu**: erken bitiş / gereksiz uzamayı azalt → `pad_loss_weight` + `EOS-stop`

### Kod Değişiklikleri
- Yeni test config:
  - `configs/lanelm/lanelm_v4_culane_test_v48_overfit32_presence_pad.py`
  - `use_presence_filter=True`, `presence_threshold=0.35`
  - `enable_eos_stop=True` (guard’lar ile)
- Yeni train+test runner script:
  - `tools/run_v48_presence_pad_train_test.sh`

### Neden bu çözüm?
V47 bulguları:
- `use_presence_filter=False` iken lane sayısı genelde 3–4’e kilitleniyor → FP yüksek.
- Pad/EOS sınırı öğrenilmediğinde lane uzunluğu kararsız → IoU düşer.

### Reproducible test (kalıntı riskini sıfırla)
`tools/test_lanelm_runner.py` artık:
- `result_dir`’i her zaman `WORK_DIR/predictions` yapar
- `--clean-preds` ile önceki prediction kalıntılarını siler


