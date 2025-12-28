## PRD_V46 — `_repro_test_v42_fix` Analizi + Testset Görselleştirme Aracı

### 1) Problem
Kullanıcı gözlemi:
- “Train/checkpoint değişmediği halde test her seferinde başka değer veriyor.”
- Repro runner ile yapılan run: `work_dirs/_repro_test_v42_fix/20251228_015611` önceki run’lardan **daha kötü** çıktı.

### 2) Repro Run Sonucu (20251228_015611)
Dosyalar:
- `work_dirs/_repro_test_v42_fix/20251228_015611/20251228_015611.json`
- `work_dirs/_repro_test_v42_fix/20251228_015611/vis_data/config.py`

Konfig:
- `ckpt_path='work_dirs/v42_overfit32_prompt2_fix/lanelm_v4_best.pth'`
- `use_prompting=True`
- `use_presence_filter=False`
- `smooth=False`
- `use_hr=True`
- `result_dir='work_dirs/_repro_test_v42_fix/predictions'`
- `use_parallel=False` (metric)

Metrik:
- F1@0.1 = **0.4254**
- F1@0.5 = **0.0274**

IoU=0.5 log (özet):
- TP=8, FP=368, FN=199 → hem precision hem recall çok düşük

### 3) Kritik Bulgular
#### 3.1) Bu run artık “gerçek” performansı gösteriyor olabilir
Önceden birçok config sabit `result_dir` kullanıyordu ve prediction dosyaları run’lar arasında overwrite/mix olabiliyordu.
Repro runner ile:
- `result_dir` artık `WORK_DIR/predictions` altında izole
- `--clean-preds` kullanıldığında stale karışma riski sıfırlanır

Bu yüzden “daha kötüye gitti” hissi, gerçekte önceki metriklerin farklı prediction’lardan geliyor olması ihtimalini yükseltir.

#### 3.2) Prediction dosyası istatistikleri (FP baskısı)
`work_dirs/_repro_test_v42_fix/predictions`:
- 100/100 dosya var, empty yok
- lane_count dağılımı:
  - 77 dosyada 4 lane
  - 22 dosyada 3 lane
  - 1 dosyada 2 lane

`use_presence_filter=False` olduğu için model genelde 4 lane üretmeye eğilimli → FP artışı ve strict IoU’da F1 düşüşü ile uyumlu.

### 4) Yeni İhtiyaç: Test Set Görsel Çıktı
Metrik tek başına yeterli değil; hangi hata tiplerinin (offset, drift, extra lanes, gap) baskın olduğunu görmek gerekiyor.

Bu amaçla yeni bir script eklendi:
- `tools/visualize_culane_pred_vs_gt.py`

Ne yapıyor?
- `list_path`’teki her `sub_img_name` için:
  - görüntüyü `data_root/sub_img_name`’den okur
  - GT’yi `data_root/sub_img_name.lines.txt`’den okur (varsa)
  - prediction’ı `pred_dir/sub_img_name.lines.txt`’den okur (varsa)
  - overlay:
    - GT: yeşil
    - PRED: kırmızı
  - `out_dir`’e `.png` olarak kaydeder (klasör yapısını korur)

### 5) Çalıştırma Komutu (Görselleştirme)

```bash
cd /home/alki/projects/clrernetlm && \
/home/alki/miniconda3/envs/clrernet/bin/python tools/visualize_culane_pred_vs_gt.py \
  --data-root dataset \
  --list-path dataset/list/test_100.txt \
  --pred-dir work_dirs/_repro_test_v42_fix/predictions \
  --out-dir work_dirs/_repro_test_v42_fix/vis_overlays \
  --max-samples 50 \
  --start-idx 0 \
  --thickness 2
```

Beklenen çıktı:
- `work_dirs/_repro_test_v42_fix/vis_overlays/.../*.png`


