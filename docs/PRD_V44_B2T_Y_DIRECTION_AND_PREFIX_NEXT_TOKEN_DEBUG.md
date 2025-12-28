## PRD_V44 — Bottom→Top y-direction + Prefix→Next Token Sanity Check

Bu PRD, V43’ten sonra yapılan iki kritik adımı dokümante eder:
- **B2T (bottom_to_top) y-direction**: prompting’i causal decoder ile uyumlu hale getirme deneyi
- **Prefix→Next Token sanity**: AR drift’in başladığı ilk timestep’i “minimal, deterministik” şekilde tespit etme

---

### 1) Problem / Motivasyon

#### A) Prompt2 neden çalışmıyor? (kanıt)
`work_dirs/_debug_prompt_t_indices/20251226_232329/20251226_232329.log`:
- CLRNet keypoint’leri normalize (0..1) ve lane’in altına yakın:
  - `Y range: [0.9084, 1.0000]`
- Fixed-y grid’e map edilince prompt:
  - `prompt_t_indices=[33, 34]`

**Sonuç:** Mevcut dizilim `top_to_bottom` iken `t=0` üst taraftır. Causal self-attn sebebiyle geç gelen prompt (t≈33/34) erken adımları yönlendiremez → prompt “refine” etkisi beklenen gibi çalışmayabilir, hatta override ile zararlı olabilir.

#### B) Prompt=0 iken bile kötü AR olabilir (exposure bias)
Bu doğru bir gözlem: TF’de çok iyi olup AR’de drift yaşanması “exposure bias” kategorisindedir.
Bu PRD’de bu drift’i nerede başladığını ölçmek için **prefix-next-token** debug aracı eklenmiştir.

---

### 2) Yapılan Değişiklikler

#### 2.1) Tokenizer y-direction flag
Dosya:
- `libs/models/lanelm/tokenizer.py`

Değişiklik:
- `LaneTokenizerConfig.y_direction` eklendi:
  - `top_to_bottom` (default)
  - `bottom_to_top`
- `_compute_sample_ys()` artık bu yöne göre sample_ys üretir.

#### 2.2) Training script: y-direction CLI + checkpoint config’a yazma
Dosya:
- `tools/train_lanelm_v4_fixed.py`

Değişiklik:
- `--y-direction {top_to_bottom,bottom_to_top}` eklendi
- `LaneTokenizerConfig(..., y_direction=args.y_direction)` ile train-time tokenizasyon yönü kontrol edildi
- Checkpoint `config` içine `y_direction` kaydedildi (debug araçları doğru tokenizer ile açabilsin diye)

#### 2.3) TF vs AR visualizer: checkpoint y_direction okuma
Dosya:
- `tools/visualize_tf_vs_ar.py`

Değişiklik:
- Tokenizer init’te `y_direction=cfg.get("y_direction","top_to_bottom")` eklendi

#### 2.4) Yeni debug aracı: Prefix → Next Token Sanity Check
Dosya:
- `tools/debug_prefix_next_token.py`

Ne ölçüyor?
- Tek bir sample için, lane-slot bazında:
  - **Teacher-prefix next-token**: GT prefix ver → `t` pozisyonundaki “next token” doğru mu?
  - **Greedy AR**: kendi tahminleriyle ilerle → ilk hata timestep’i nerede?

Çıktı:
- `TF-prefix next-token acc` ve `first_mismatch_t`
- `AR greedy acc` ve `first_mismatch_t`
- İsteğe bağlı json çıktı (`--save-json`)

#### 2.5) V44 test config
Dosya:
- `configs/lanelm/lanelm_v4_culane_test_v44_overfit32_prompt2_b2t.py`

Amaç:
- MMEngine test tarafında `tokenizer_cfg.y_direction='bottom_to_top'` ile b2t deneyi.

---

### 3) Çalıştırma Komutları (Train + Test + Debug)

#### 3.1) V44 Train (overfit32, prompt2, b2t)
Komut:

```bash
cd /home/alki/projects/clrernetlm && \
/home/alki/miniconda3/envs/clrernet/bin/python tools/train_lanelm_v4_fixed.py \
  --list-path dataset/list/train_2k.txt \
  --overfit-size 32 \
  --epochs 100 \
  --num-pseudo-points 2 \
  --y-direction bottom_to_top \
  --x-embedding-scale 1.0 \
  --lane-embedding-boost 1.0 \
  --ss-max-prob 0.2 \
  --ar-rollout-max-weight 0.05 \
  --ar-rollout-min-weight 0.02 \
  --presence-weight 0.0 \
  --pad-loss-weight 1.0 \
  --work-dir work_dirs/v44_overfit32_prompt2_b2t \
  --device cuda | tee work_dirs/v44_overfit32_prompt2_b2t/train.log
```

Not:
- Bu eğitim bittiğinde checkpoint: `work_dirs/v44_overfit32_prompt2_b2t/lanelm_v4_best.pth`

#### 3.2) V44 Test (CULane test_100)

```bash
cd /home/alki/projects/clrernetlm && \
/home/alki/miniconda3/envs/clrernet/bin/python tools/test_lanelm_runner.py \
  configs/lanelm/lanelm_v4_culane_test_v44_overfit32_prompt2_b2t.py \
  --work-dir work_dirs/v44_test100_overfit32_prompt2_b2t
```

#### 3.3) Prefix → Next Token Debug (minimal sanity)

```bash
cd /home/alki/projects/clrernetlm && \
/home/alki/miniconda3/envs/clrernet/bin/python tools/debug_prefix_next_token.py \
  --lanelm-ckpt work_dirs/v44_overfit32_prompt2_b2t/lanelm_v4_best.pth \
  --list-path dataset/list/train_2k.txt \
  --sample-idx 0 \
  --device cuda \
  --save-json work_dirs/_debug_prefix_next_token/v44_sample0.json
```

---

### 4) Beklenen Sonuç / Başarı Kriteri
- **B2T prompting** ile prompt_t indeksleri küçükleşmeli (örn. t=0..10 bandına gelmeli).
- `debug_prefix_next_token.py` çıktısında:
  - TF-prefix first_mismatch daha geç başlamalı
  - AR first_mismatch daha geçe kaymalı (drift azalmalı)
- CULane test_100’de F1@0.5 baseline üstüne çıkması beklenir (özellikle FN/empty azalması).


