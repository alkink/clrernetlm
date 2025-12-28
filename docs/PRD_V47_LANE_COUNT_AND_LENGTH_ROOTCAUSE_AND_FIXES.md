## PRD_V47 — Şerit Sayısı + Şerit Uzunluğu Problemi: Kök Nedenler ve Fix Planı

### 1) Semptom (kullanıcı görselleri ile uyumlu)
- Şeritler genel olarak yola “oturuyor” (geometri kaba olarak doğru)
- Ancak:
  - **Şerit sayısı** fazla/kararsız (çoğu karede 3–4)
  - **Şerit uzunluğu** hatalı (bazı lane’ler gereksiz uzuyor / bazıları erken bitiyor)
- Bu durum CULane strict IoU (0.5) skorunu dramatik düşürür:
  - ekstra lane → FP artar
  - yanlış uzunluk/segment → IoU düşer → TP azalır + FN artar

### 2) Kanıt 1 — Repro run metrikleri ve FP baskısı
`work_dirs/_repro_test_v42_fix/20251228_015611`:
- F1@0.5 ≈ **0.027**
- log: IoU=0.5 için **TP=8 FP=368 FN=199** → FP aşırı yüksek, TP çok düşük.

`work_dirs/_repro_test_v42_fix/predictions` istatistiği:
- empty dosya yok
- lane_count: 77 dosyada 4, 22 dosyada 3, 1 dosyada 2

Bu, **presence/lane-slot filtresi olmadığı** durumda beklenen bir FP paterni.

### 3) Kanıt 2 — Prefix→Next Token debug: erken drift / generalization
`tools/debug_prefix_next_token.py` ile test_100 içindeki GT’li bir örnek (idx=31):
- TF-prefix next-token acc çok düşük (slot0 ~0.05)
- İlk mismatch çoğu slotta **t=8**’de başlıyor
- Bazı timesteplerde model GT yerine **0 (pad/EOS)** üretiyor

Bu, sadece “exposure bias” değil; aynı zamanda **modelin testte generalize edemediğine**
ve/veya **pad/EOS sınırlarını doğru öğrenemediğine** işaret eder.

### 4) Kök Nedenler (2 eksen)

#### A) Şerit sayısı (FP) — presence head / filtre eksikliği
- `use_presence_filter=False` iken model çoğu zaman 3–4 lane üretir.
- Presence head eğitilmediyse (presence_weight=0) filtreyi açmak da güvenilir olmaz.

**Kalıcı çözüm:**
- training: `--presence-weight 1.0` (veya sweep)
- test: `use_presence_filter=True` + threshold tuning

#### B) Şerit uzunluğu (segment) — pad/EOS sınırı + “scattered tokens”
- Model bazen yanlış yerlerde `x=0` (erken bitiş) veya dağınık non-zero tokenlar (gereksiz uzama) üretebilir.
- Strict IoU’da bu “segment” hatası çok pahalıdır.

**Kalıcı çözüm:**
- training: `--pad-loss-weight 1.0` (gerekirse 2.0) + EOS-stop kullanımı
- ek: (opsiyonel) “validity” head / start-end head (daha büyük mimari değişiklik)

**Hızlı (eğitimsiz) stabilizasyon:**
- decode-time “contiguous run” seçimi: non-pad tokenların en uzun ve bottom’a en yakın segmentini tut, kalanlarını pad yap.

### 5) Uygulanan Fix (V47): contiguous-run postprocess
Dosya:
- `libs/models/detectors/lanelm_detector.py`

Yeni decode_cfg bayrakları:
- `contiguous_run=True|False` (default False)
- `contiguous_min_len` (default 2)

Yeni test config:
- `configs/lanelm/lanelm_v4_culane_test_v47_repro_v42fix_contiguous.py`

Amaç:
- “tek lane içinde” dağınık tokenlardan doğan gereksiz uzamayı azaltmak
- valid_count düşerse lane drop ile FP’yi azaltmak


