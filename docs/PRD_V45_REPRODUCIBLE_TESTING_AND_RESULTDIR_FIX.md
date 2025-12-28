## PRD_V45 — Reproducible Testing: result_dir overwrite/mixing ve deterministik runner

### 1) Problem
Son dönemde “train değişmiyor ama test metrikleri değişiyor” semptomu görüldü.

Bu repo’da `CULaneMetric` prediction dosyalarını `test_evaluator.result_dir` içine yazar ve sonra aynı klasörden tekrar okuyarak metrik hesaplar.

Kritik risk:
- Birçok test config’i **aynı statik `result_dir`** kullanıyordu (örn. `work_dirs/v42_test100_overfit32_prompt2_fix/predictions`).
- Farklı `--work-dir` ile test koşsan bile prediction’lar aynı klasöre yazıldığı için:
  - run’lar birbirini overwrite eder
  - görseller/çıktılar karışır
  - hangi metrik hangi prediction’a ait takibi zorlaşır

Bu durum kullanıcı tarafında “aynı deneyi tekrar koşunca farklı çıktı alıyorum” algısını güçlendirir.

### 2) Kanıt
Repo içinde birden fazla run’ın `vis_data/config.py` çıktısında aynı result_dir görünüyordu:
- `work_dirs/_debug_prompt_t_indices/*/vis_data/config.py`
- `work_dirs/v42_test100_overfit32_prompt2_fix_sorted/*/vis_data/config.py`

Hepsinde:
- `result_dir='work_dirs/v42_test100_overfit32_prompt2_fix/predictions'`

### 3) Çözüm
`tools/test_lanelm_runner.py` güncellendi:
- Her test koşusunda metric `result_dir` **otomatik olarak** `WORK_DIR/predictions` olacak şekilde override edilir.
- İsteğe bağlı:
  - `--clean-preds`: prediction klasörünü testten önce siler (stale/mix riskini sıfırlar)
  - `--seed` + `--deterministic`: deterministik koşum için
  - `--no-parallel-metric`: metric’in multiprocessing’ini kapatır (debug/determinism)

Dosya:
- `tools/test_lanelm_runner.py`

### 4) Yeni Kullanım (önerilen)

```bash
cd /home/alki/projects/clrernetlm && \
/home/alki/miniconda3/envs/clrernet/bin/python tools/test_lanelm_runner.py \
  configs/lanelm/lanelm_v4_culane_test_v42_overfit32_prompt2_fix.py \
  --work-dir work_dirs/_repro_test_v42_fix \
  --clean-preds \
  --seed 0 \
  --no-parallel-metric
```

Beklenen:
- Aynı checkpoint + aynı config ile tekrar tekrar koşunca **aynı metrikler**.
- Prediction dosyaları her zaman ilgili `work_dir` altında tutulur, karışmaz.


