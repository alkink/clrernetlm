# V7: Kalıcı Çözümler Tamamlandı

## Yapılan Değişiklikler

### 1. nbins_x: 200 → 800 (PDF Standard) ✅

**Kök Neden:** Zigzagging'in ana nedeni tokenization granularity'nin çok kaba olmasıydı (200 bins = 4px per bin).

**PDF Referansı:**
- Line 570: "800 nbins and 100 training epochs"
- Line 1641: "nbins=800 aligns with our intuition"
- Table 7: `800 / 1280` → F1=97.64 (en iyi sonuç)

**Değişiklikler:**
1. **Training Script** (`tools/train_lanelm_v4_fixed.py`):
   - `nbins_x = 800` (line 215)

2. **Test Config** (`configs/lanelm/lanelm_v4_culane_test.py`):
   - `lanelm_cfg.nbins_x = 800`
   - `tokenizer_cfg.nbins_x = 800`

3. **Tokenizer Default** (`libs/models/lanelm/tokenizer.py`):
   - `LaneTokenizerConfig.nbins_x = 800` (zaten default)

4. **Model** (`libs/models/lanelm/model.py`):
   - Model zaten `nbins_x` parametresini alıyor, training script'ten geçiliyor

**Beklenen Etki:**
- Granularity: 800 / 800 = **1px per bin** (PDF standard)
- Zigzagging azalması bekleniyor
- Daha smooth lane predictions

### 2. Prompting Strategy (CLRNet'ten İlk 2 Keypoint) ✅

**PDF Referansı:**
- Line 497-499: "A regression network is employed to provide the two initial keypoints, for each lane. LaneLM is responsible for completing the remaining keypoints."
- Table 3: 2-kp prompting +3-6% F1 artışı sağlıyor

**Implementasyon:**
1. **CLRerNet Model Build** (`libs/models/detectors/lanelm_detector.py`):
   - `LaneLMDetector.__init__`: CLRerNet model'i (backbone + neck + head) build ediliyor
   - CLRerNet head weights checkpoint'ten yükleniyor
   - Model freeze ediliyor

2. **Keypoint Extraction** (`LaneLMDetector.predict`):
   - CLRerNet model'i ile lane predictions alınıyor
   - Her lane için ilk 2 keypoint extract ediliyor
   - Keypoint'ler resized space'e (800x320) transform ediliyor
   - Tokenize ediliyor (`initial_x_tokens`, `initial_y_tokens`)

3. **Autoregressive Decode Integration** (`autoregressive_decode`):
   - `initial_x_tokens` ve `initial_y_tokens` parametreleri eklendi
   - İlk 2 timestep'te CLRNet keypoint'leri kullanılıyor (prompt olarak)
   - Sonraki timestep'lerde normal autoregressive decode devam ediyor

**Beklenen Etki:**
- Daha iyi başlangıç noktaları (CLRNet'ten gelen)
- Error accumulation azalması
- F1@0.5 artışı (+3-6% PDF'ye göre)

## Sonraki Adımlar

1. **Model'i 800 bins ile yeniden eğit:**
   ```bash
   python tools/train_lanelm_v4_fixed.py \
     --data-root dataset \
     --list-path dataset/list/train_100.txt \
     --overfit-size 1 \
     --epochs 100
   ```

2. **Test config'i güncelle:**
   - Yeni checkpoint path'i güncelle (`lanelm_v4_culane_test.py`)

3. **Test et ve sonuçları karşılaştır:**
   - F1@0.5 artışı bekleniyor
   - Zigzagging azalması bekleniyor
   - Daha smooth predictions bekleniyor

## Notlar

- **Checkpoint Uyumluluğu:** Mevcut checkpoint (`lanelm_v4_best.pth`) 200 bins ile eğitilmiş. 800 bins ile yeniden eğitim gerekli.
- **Prompting Strategy:** `use_prompting=True` (default) ile aktif. `decode_cfg.use_prompting=False` ile kapatılabilir.
- **Memory Kullanımı:** CLRerNet model'i ek memory kullanıyor, ancak freeze edildiği için training'de sorun yok.

## Dosya Değişiklikleri

1. `tools/train_lanelm_v4_fixed.py`: `nbins_x = 800`
2. `configs/lanelm/lanelm_v4_culane_test.py`: `nbins_x = 800` (her iki yerde)
3. `libs/models/detectors/lanelm_detector.py`:
   - CLRerNet model build ve keypoint extraction
   - `autoregressive_decode` prompting integration
4. `libs/models/detectors/lanelm_detector.py` (`autoregressive_decode`):
   - `initial_x_tokens` ve `initial_y_tokens` parametreleri
   - Prompting logic (ilk 2 timestep)








