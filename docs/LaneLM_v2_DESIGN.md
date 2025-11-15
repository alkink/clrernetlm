## LaneLM v2 Tasarımı ve Mevcut Durum

Bu doküman, `LaneLM: Lane Detection as Language Modeling` makalesindeki fikirleri bu repoda nasıl uyguladığımızı ve hâlâ eksik kalan noktaları özetler. Amaç, yapılan her değişikliği ve makaleden sapmaları açık ve dürüst biçimde ortaya koymaktır.

---

## 1. Makaledeki Temel Fikirler (Özet)

- **Lane tokenizasyonu (3.1 Lane Representation)**
  - Her şerit: \(L = \{(x_1, y_1), ..., (x_T, y_T)\}\).
  - \(y_t = \frac{H}{T} \cdot t\): y koordinatları dikey eksende eşit aralıklı örneklenir.
  - Her zaman adımında iki token: \((x_t, t)\).
    - \(x_t \in [0, nbins)\): x quantize.
    - y tarafı aslında adım index’i \(t\) (0..T-1), T padding/EOS.
  - 0 (x için) ve T (y için) padding/EOS token’larıdır.
  - Embedding:
    \[
      e_t = E_y(y_t) + E_x(x_t) + PE_{keypoint}(t)
    \]

- **Probabilistic rollouts (3.2)**
  - LD, şartlı dil modeli olarak formüle edilir:
    \[
      p_\theta(A_1,...,A_T | X_v) = \prod_t p_\theta(A_t | A_{<t}, X_v)
    \]
    \[
      p_\theta(A_t | A_{<t}, X_v) = \prod_n p_\theta(a^n_t | A_{<t}, X_v)
    \]
  - Aynı zamanda bir görüntüdeki her lane \(n\) için t anındaki keypointler koşullu bağımsız kabul edilir.

- **Görsel encoder (3.3 Network Architecture)**
  - CNN + FPN: \(f(X_v) \to \{F_0, F_1, F_2\}\).
  - Her seviye \(F_i\) patch’lere bölünüp ViT benzeri şekilde embed edilir:
    \[
      L_i = E_v(F_i) + PE_{vision}(H_i, W_i) + LE(i)
    \]
  - Tüm seviyeler tek bir token dizisi olarak decoder’a verilir.

- **Training (3.4 Training and Inference)**
  - Teacher (CLRNet [6]) pseudo label üretir: \(L^q\).
  - GT ile eşleştirilir: (L^q_i, L^gt_j) → çiftler, Hungarian matching ile.
  - Multi-turn VQA: \(S = (L^1_q \circ L^1_{gt}, ..., L^N_q \circ L^N_{gt}, X_v)\).
  - Loss (Eq. 11):
    \[
      \max \sum_t \big(\log P(x_t | x_{<t}, X_v) + \log P(y_t | y_{<t}, X_v)\big)
    \]
  - Inference:
    - Greedy argmax:
      - \(x_t = \arg\max P(x_t | x_{<t}, X_v)\)
      - \(y_t = \arg\max P(y_t | y_{<t}, X_v)\)
    - EOS: x=0 veya y=T → sequence sonu.

---

## 2. Bu Repodaki LaneLM Bileşenleri

### 2.1 Tokenizer ve Embedding

- `libs/models/lanelm/tokenizer.py`
  - `LaneTokenizerConfig`:
    - `img_w=800, img_h=320, num_steps=40, nbins_x=800, pad_token_x=0, pad_token_y=-1`.
  - `LaneTokenizer`:
    - Y eksenini `num_steps = T` boyunca eşit aralıklı örnekler (`_compute_sample_ys`).
    - `encode_single_lane`:
      - Spline ile x(y) fit eder.
      - Her y için x’i [1, nbins_x-1] aralığına quantize eder; 0 padding.
      - y token’ı time-step index’i t; padding için T kullanılır.
    - `decode_single_lane`:
      - Token’ları tekrar continuous (x,y) koordinatlarına çevirir.

- `libs/models/lanelm/model.py`
  - `KeypointEmbedding`:
    - `x_embedding`, `y_embedding`, `pos_embedding` → \(E_x + E_y + PE\), Eq. (2) ile uyumlu.

### 2.2 Görsel Token Encoder (VisualTokenEncoder)

- `VisualTokenEncoder` (`libs/models/lanelm/model.py`)
  - Girdi: FPN seviyeleri `[F0, F1, F2]` (CLRerNetFPN out: 3 seviye).
    - Her biri: `(B, C_i, H_i, W_i)`; C_i LaneLM v2’de `(64,64,64)` olarak konfigüre edildi.
  - İş akışı:
    - Her seviye için:
      - `(B, C, H, W)` → `(B, H*W, C)` → `Linear(C → D)` → patch embedding.
      - Level embedding (`LE(i)`) eklenir.
    - Ardından tüm seviyeler concat: `(B, N_total, D)`; N_total ≈ 5250.
    - 1D positional embedding (`pos_embedding`) ile `PE_vision` yerine geçen pratik bir çözüm kullanılır.
  - `max_tokens = 8192`:
    - CULane için 40×100 + 20×50 + 10×25 = 5250 token’ı güvenle kapsayacak bir üst sınır.

- `LaneLMModel.encode_visual_tokens(feats)`:
  - `VisualTokenEncoder` ile FPN feature’larını LaneLM token’larına çevirir.

### 2.3 Decoder ve Head

- `LaneLMDecoderLayer` + `LaneLMDecoder`:
  - Causal self-attention + cross-attention (visual tokens) + FFN, LayerNorm.
  - Causal mask `triu` ile üretiliyor; geleceğe bakışı engelliyor.

- `LaneLMHead`:
  - `proj_x`: D → nbins_x
  - `proj_y`: D → max_y_tokens

- `LaneLMModel.forward`:
  - Girdi:
    - `visual_tokens`: (B, N, D), `encode_visual_tokens` çıktısı.
    - `x_tokens`, `y_tokens`: (B, T).
  - Çıkış: `logits_x` (B,T,nbins), `logits_y` (B,T,max_y_tokens).
  - `visual_encoder` ile kurulan modelde `visual_tokens` zaten D boyutunda kabul edilir; ek projeksiyon yok.

---

## 3. LaneLM v2 Training Script (tools/train_lanelm_culane_v2.py)

Bu script, makaledeki LaneLM training mantığını mümkün olduğunca yakından implemente eder.

### 3.1 DataLoader ve Pipeline

- `build_culane_lanelm_dataloader_v2(...)`
  - `CulaneDataset` + `train_al_pipeline`:
    - Aynı CULane augmentasyonları (Crop, Resize, Albumentations) tekrar kullanılır.
    - `PackCLRNetInputs` **kullanılmaz**; `gt_points` ham halde bırakılır.
  - `collate_lanelm_batch_v2`:
    - `img` → float32 /255 ile tensor (B,3,H,W).
    - `gt_points` → her görüntü için list[list[x0,y0,...]] olarak korunur.
    - `sub_img_name`, `ori_shape`, `img_shape` meta bilgileri tutulur.

### 3.2 Frozen CLRerNet Backbone + FPN

- `build_frozen_clrernet_backbone(...)`:
  - `init_detector` ile CLRerNet yüklenir.
  - `model.backbone` ve `model.neck` (`CLRerNetFPN`) `requires_grad=False` yapılır.

- `extract_pyramid_feats(model, imgs)`:
  - `model.extract_feat(imgs)` → 3 seviyeli FPN feature listesi.

### 3.3 LaneLM v2 Model

- `LaneLMHyperParams`:
  - CULane için default:
    - `num_points=40`, `nbins_x=800`, `max_lanes=4`, `img_w=800`, `img_h=320`.
    - `embed_dim=256`, `num_layers=4`, `num_heads=8`, `ffn_dim=512`.

- `build_lanelm_model_v2(hparams, visual_in_channels)`:
  - `visual_in_channels=(64,64,64)`:
    - CLRerNetFPN’in çıkış kanalları.
  - `max_y_tokens = T+1`, `max_seq_len = T * max_lanes * 2` (güvenli üst sınır).
  - `LaneLMModel` `visual_in_channels` ile kurulur → VisualTokenEncoder aktif.

### 3.4 Multi-Lane Sequence İnşası (L ≈ L_q◦L_gt)

- `build_sequences_for_image(lanes_points, tokenizer, hparams)`:
  - Her görüntü için:
    - `lanes_points` içindeki her lane:
      - Çok az noktası olanlar filtrelenir.
      - `LaneTokenizer.encode_single_lane` ile `(x_tokens, y_tokens)` (T uzunluklu) üretilir.
    - En fazla `max_lanes` lane alınır.
    - Sequence:
      - L1, L2, ..., LN concat edilerek uzun bir dizi oluşturulur: `(L1; L2; ...; LN)`.
      - `max_seq_len = T * max_lanes * 2` sınırına göre truncate/pad yapılır.
  - Not:
    - Şu an bu sequence *fikir olarak* `L_q◦L_gt` yerine `L_gt◦L_gt` gibi davranır; yani pseudo label’ı henüz teacher’dan ayrı üretmiyoruz.

### 3.5 Loss ve Training Loop

- `lane_lm_loss_v2`:
  - `CrossEntropyLoss(ignore_index=pad_token_x)` ve `ignore_index=pad_token_y` (T).
  - Toplam loss: 0.5 * (loss_x + loss_y), Eq. (11) ile uyumlu.

- `train_one_epoch_v2(...)`:
  - Adımlar:
    1. `extract_pyramid_feats` → FPN feature’ları.
    2. `encode_visual_tokens(feats)` → `(B, N_total, D)` görsel tokenlar.
    3. Her görüntü için `build_sequences_for_image`:
       - GT lane’lerden multi-lane token sequence (x_tokens, y_tokens) çıkarılır.
    4. Teacher forcing:
       - `x_in[:,0]=PAD`, `x_in[:,1:]=x_tokens[:,:-1]`.
       - `y_in` için benzer shift; ilk token padding (=T).
    5. `LaneLMModel(visual_tokens, x_in, y_in)` → logits.
    6. Loss hesaplanır, backward + optimizer step.

- `main()`:
  - CLI arg’ler: config, checkpoint, data-root, train-list, diff-file, epochs, batch-size, lr, device, work-dir.
  - Checkpoint:
    - `lanelm_culane_dla34_v2_epoch{epoch}.pth`:
      - `model_state_dict`, `optimizer_state_dict`, hyperparam config.

---

## 4. Test (tools/test_lanelm_culane.py) – v1/v2 Durumu

Bu script ilk etapta v1 training için yazıldı, fakat şu anda:

- CULane test seti (`dataset/list/test.txt`) + `COLaneMetric` kullanıyor:
  - `CULaneMetric.compute_metrics(...)` + `.lines.txt` yazma + IoU/F1 hesaplama **CLRerNet ile bire bir aynı**.
- LaneLM inference:
  - `extract_visual_tokens` ile CLRerNet FPN’den `(B,N,D)` token alıyor.
  - `autoregressive_decode_tokens`:
    - EOS kullanan otoregresif decode:
      - Her step’te `logits_x`, `logits_y` → argmax veya temperature sampling.
      - x=0 veya y=T geldiğinde sequence bitiyor.
    - Parametre `max_lanes` ile birden fazla sequence üretmeye hazır; şu an `max_lanes=1` geçiyoruz.
  - Her token dizisi `LaneTokenizer.decode_single_lane` ile 800×320 uzayında koordinatlara, ardından `coords_to_lane` ile orijinal CULane koordinat sistemine (1640×590 + crop) geri taşınıyor.
  - Bu lane listeleri `CULaneMetric`’e veriliyor, evaluation CLRerNet ile aynı prosedürü izliyor.

LaneLM v2 training’le tam uyumlu test için, `LaneLMModel`’i `encode_visual_tokens` ile kullanılan yeni görsel encoder’la da bağlayacak bir `test_lanelm_culane_v2.py` eklenmesi planlanmıştır; şu anki script v1/v2 karışık bir durumda, ama metric tarafı sağlamdır.

---

## 5. Makaleye Göre Eksik Kalan Noktalar (Tavizler)

Dürüstçe, hâlâ eksik olan / yaklaşık yapılan kısımlar:

1. **Pseudo label L_q ve multi-turn VQA tamamen uygulanmadı**
   - Makalede:
     - Teacher CLRNet pseudo keypoint label üretir → L_q.
     - GT ile Hungarian matching yapılır → (L_q, L_gt) çiftleri.
     - Multi-turn sequence: \(S = (L^1_q \circ L^1_{gt}, ..., L^N_q \circ L^N_{gt})\).
   - Bizim v2:
     - Şimdilik doğrudan GT lane’lerden multi-lane sequence oluşturuyoruz.
     - Yani L_q ≈ L_gt kabul ediyoruz, p(L|X_v) öğreniyoruz ama öğretmenden ayrı bir query dizisi üretmiyoruz.
   - Etki:
     - Promptlu/interactive kullanım ve teacher’ın hatalarını düzeltme yönü eksik.
     - Buna rağmen, LaneLM’nin “p(L|X_v)” dil modeli mantığı büyük ölçüde korunuyor; multi-turn VQA entegre edilirse özellikle prompt senaryolarında ek kazanç beklenir.

2. **Prompt stratejileri (4-point prompts, VLM/human feedback) implement edilmedi**
   - Makalede 3 prompting stratejisi var:
     - Öğretmenden gelen başlangıç keypointleri.
     - Pseudo label’lar soru, GT cevap gibi.
     - Manuel/human veya VLM’den gelen keypoint prompt’ları.
   - Mevcut kod:
     - Prompt parametreleri için temel yapı var (tokenizer, teacher forcing), fakat VQA tarzı prompt/girdi henüz tamamlanmadı.

3. **Cache / KV cache optimizasyonları yok**
   - Makale, FPNe seviye cache’leme ve KV-cache’ten bahsediyor.
   - Bizim implementation:
     - Decoder standard PyTorch MultiheadAttention kullanıyor; KV-cache yok.
     - Bu, accuracy’yi etkilemez; sadece inference hızında taviz anlamına gelir.

4. **Eğitim süresi ve batch size**
   - Makalede:
     - 100 epoch, batch size 128, teacher CLRNet ile birlikte.
   - Bizim v2:
     - Şu an için deneysel run’lar 3–x epoch ve daha küçük batch size ile başlatıldı (sanity check).
     - Maksimum performans hedefleniyorsa, training schedule’ı makaledeki seviyeye çekmek gerekir.

---

## 6. Genel Değerlendirme

Bu repo şu anda:

- LaneLM makalesinin **temel matematiksel formülasyonunu** (tokenizasyon, görsel encoder + patch embedding, decoder-only LM, CE loss, EOS ile otoregresif inference) oldukça yakın bir şekilde hayata geçiriyor:
  - Tokenizer ve embedding → Eq. (2).
  - VisualTokenEncoder → Eq. (7)-(8).
  - Loss fonksiyonu → Eq. (11).

- **Eksik olan / yaklaşık uygulanan kısımlar**:
  - Teacher tabanlı pseudo label L_q ve multi-turn Q/A yapısı (S dizisi).
  - Prompt tabanlı interaktif kullanım senaryoları.
  - KV-cache ve diğer hız optimizasyonları.

Bu noktadan sonraki doğal adımlar:

1. Training tarafına tam teacher L_q + Hungarian matching + (L_q◦L_gt) multi-turn sequence eklemek.
2. LaneLM v2 için ayrı bir test script’i (`tools/test_lanelm_culane_v2.py`) ile EOS + multi-lane decoding’i kullanarak CULaneMetric ile performansı ölçmek.
3. Prompt senaryolarını (özellikle 4 keypoint prompt) kodlamak ve CULane’de corner case split’lerinde farkı incelemek.

Bu doküman, hangi noktada makalenin birebir izlenip hangi noktalarda yaklaşık kaldığımızı açıkça kayıt altına almak içindir; böylece ileride yapılacak geliştirmeler net bir hedefe göre planlanabilir.

