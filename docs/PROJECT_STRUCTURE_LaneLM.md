## Genel Bakış

Bu belge, `clrernetlm` deposunun mimarisini, CLRerNet tabanını ve bunun üzerine inşa edilen LaneLM tarzı bileşenleri ayrıntılı olarak açıklar. 
Amaç, kodu çalıştırmadan önce sistemin tamamını zihinde netleştirmek ve LaneLM makalesindeki fikirlerin bu projede hangi somut sınıflar ve akışlarla karşılandığını açıkça ortaya koymaktır.

Bu analiz üç ana eksene dayanır:
- CLRerNet çekirdeği (backbone, FPN, anchor tabanlı head, LaneIoU kaybı).
- CULane veri seti ve veri işleme pipeline’ı.
- LaneLM benzeri dil-modeli başlık (tokenizer, Transformer decoder, training script).

## Üst Düzey Dizin Yapısı

- `configs/`
  - `clrernet/culane/`:
    - `base_clrernet.py`: Model mimarisinin temel tanımı (backbone, neck, head, loss ve train/test cfg).
    - `dataset_culane_clrernet.py`: CULane veri pipeline’ı ve dataloader konfigleri.
    - `clrernet_culane_dla34.py`, `clrernet_culane_dla34_ema.py`: CULane için DLA34 konfigleri (EMA’lı / normal).
- `libs/`
  - `models/`: Tüm model bileşenleri (backbone, neck, dense head, LaneLM, loss’lar, vb).
  - `datasets/`: CULane dataset sınıfı ve pipeline transformları.
  - `core/`: Anchor generator, assignment / cost fonksiyonları, hook’lar.
  - `utils/`: Lane yardımcı fonksiyonları ve görselleştirme.
- `tools/`
  - `train.py`: MMDetection tarzı CLRerNet eğitimi (anchor tabanlı head).
  - `test.py`, `speed_test.py`: Eval ve hız testi.
  - `train_lanelm_culane.py`: CLRerNet backbone üstüne LaneLM-style başlık eğiten özel script (dil modeli kısmı).
- `docs/`
  - `DATASETS.md`, `INSTALL.md`: Veri seti ve kurulum dokümantasyonu.
  - `PROJECT_STRUCTURE_LaneLM.md`: Bu analiz.

Bu yapı, mevcut MMDetection ekosistemiyle uyumlu klasik bir anchor tabanlı lane detection modeli (CLRerNet) ile, tamamen bağımsız bir PyTorch training loop üzerinden eğitilen dil-modeli esinli LaneLM başlığını yan yana barındırıyor.

## CLRerNet Çekirdeği

### Model Konfigürasyonu (`configs/clrernet/base_clrernet.py`)

`model` dict’i MMDetection 3.x API’sine uygun şekilde tanımlanmış:

- `type="CLRerNet"`:
  - `libs/models/detectors/clrernet.py` içinde tanımlı, `SingleStageDetector`’dan türeyen dedike sınıf.
  - Sadece `forward_train` ve `predict` override ediliyor; mimari bütünlüğü backbone/neck/head’e devrediliyor.

- `data_preprocessor`:
  - `DetDataPreprocessor` ile mean/std normalizasyonu yapıyor.
  - `mean=[0, 0, 0]`, `std=[255, 255, 255]`, `bgr_to_rgb=False`: Görüntü BGR formatında, 0–255 aralığında kabul ediliyor.

- `backbone`:
  - `type="DLANet"`, `dla="dla34"`, `pretrained=True`.
  - DLA family’sinin 34 katmanlı varyantı; feature pyramid’e uygun çoklu seviye özellik üretir.

- `neck`:
  - `type="CLRerNetFPN"`, `in_channels=[128, 256, 512]`, `out_channels=64`, `num_outs=3`.
  - Backbone’un orta-yüksek seviye feature’larını 3 seviyeli bir FPN’e çevirir.

- `bbox_head`:
  - `type="CLRerHead"` (`libs/models/dense_heads/clrernet_head.py`).
  - Anahtar parametreler:
    - `anchor_generator`: `CLRerNetAnchorGenerator` ile toplam `num_priors=192`, `num_points=72`.
    - `prior_feat_channels=64`: FPN çıkış boyutuyla eşleşecek şekilde.
    - `refine_layers=3`, `sample_points=36`: Çok aşamalı anchor refinement pipeline’ı.
    - `attention=dict(type="ROIGather")`: ROI feature aggregation modülü (LaneATT/CLRNet benzeri).
    - Loss bileşenleri:
      - `loss_cls`: `KorniaFocalLoss` (alpha=0.25, gamma=2) – pozitif/negatif dengesizliğini çözmek için.
      - `loss_bbox`: `SmoothL1Loss` – anchor parametreleri ve uzunluk regrese etmek için.
      - `loss_iou`: `LaneIoULoss` – LaneIoU metrik/başarım uyumluluğu için temel yenilik.
      - `loss_seg`: `CLRNetSegLoss` – 5 sınıflı (4 lane + background) segmentation kaybı, opsiyonel.

- `train_cfg`:
  - `assigner` = `DynamicTopkAssigner` + özel `LaneIoUCost`:
    - `cls_cost` = FocalCost, `reg_cost` = DistanceCost.
    - İki farklı `LaneIoUCost`:
      - `iou_dynamick`: geniş lane_width, GIoU destekli, top-k seçim için.
      - `iou_cost`: farklı genişlik, `use_pred_start_end=True` ile refine edilmiş prior’ları hesaba katan asıl cost.

- `test_cfg`:
  - `conf_threshold=0.41`: makaledeki cross-validation’dan gelen eşik.
  - `use_nms=True`, `nms_thres=50`, `nms_topk=4`: Lane-level NMS.
  - `as_lanes=True`, `extend_bottom=True`: Çıkışlar Lane nesneleri/koordinatları olarak üretiliyor ve alt kısım doğru şekilde uzatılıyor.
  - `ori_img_w=1640`, `ori_img_h=590`, `cut_height=270`: CULane’nin kırpılmış/ölçeklendirilmiş koordinat sistemine geri dönüş için gerekli meta parametreler.

### CLRerNet Dedektörü (`libs/models/detectors/clrernet.py`)

- `CLRerNet(SingleStageDetector)`:
  - `forward_train`:
    - `x = self.extract_feat(img)` ile backbone + neck’ten çok seviyeli feature’ları çıkarır.
    - `self.bbox_head.forward_train(x, img_metas)` ile tüm loss bileşenlerini head’e delege eder.
  - `predict`:
    - `data_samples[i].metainfo["batch_input_shape"]` ayarlanır.
    - `self.bbox_head.predict(x, data_samples)` ile lane’ler üretilir.

Bu sınıf, mimarinin “pipeline glue” parçasıdır; asıl iş yükü `CLRerHead` ve backbone/neck’te.

### CLRer Head (`libs/models/dense_heads/clrernet_head.py`)

Ana fonksiyonlar ve veri akışı:

- `__init__`:
  - `anchor_generator`, `attention`, loss ve assigner nesnelerini `TASK_UTILS.build` / `MODELS.build` ile üretir.
  - Anchor parametrelerini, FPN kanallarını, strip sayısını (y ekseninde kaç satır kullanıldığı) ve sample noktalarını hesaplar.

- `pool_prior_features`:
  - Girdi: `batch_features` (B, C, H, W) ve `priors_on_featmap` (B, Np, Ns).
  - `grid_sample` ile anchor hatları boyunca ROI feature pooling yapar.
  - Çıktı: (B * Np, C, Ns, 1) – her anchor için 1D lane profilini temsil eden feature.

- `forward`:
  - Girdi: FPN çıkışları `x` (3 seviye).
  - İş akışı:
    1. `feature_pyramid` = FPN’in en yüksek çözünürlükten itibaren `refine_layers` kadar seviyesi, ters çevrilip kullanılıyor.
    2. Başlangıç priors:
       - `anchor_generator.generate_anchors` ile y seviyelerine göre x koordinatları ve anchor parametreleri üretiliyor.
    3. Refinement döngüsü (`for stage in range(self.refine_layers)`):
       - `pool_prior_features` ile anchor’lar boyunca ROI feature pooling.
       - `self.attention` (ROIGather) ile farklı FPN seviyelerinden bağlamsal lane feature’ı toplanıyor.
       - `cls` ve `reg` fully-connected katmanları ile:
         - `cls_logits` (B, Np, 2).
         - `reg` = `[dy0, dx0, dtheta, dlength, dxs...]`.
       - Anchor parametreleri güncelleniyor (`anchor_params += reg[..., :3]`).
       - Yeni anchor x’leri hesaplanıp `reg_xs` çıkarılıyor.
       - `pred_dict`:
         - `cls_logits`, `anchor_params`, `lengths`, `xs`.
       - Her stage için `predictions_list`’e ekleniyor.
       - Son stage haricinde, anchor parametreleri detach edilip bir sonraki refinement için kullanılıyor.

- `loss_by_feat`:
  - Girdi: `out_dict["predictions"]` (3 stage’lik anchor tahminleri) ve `batch_data_samples`.
  - Her stage ve her batch örneği için:
    - `self.assigner.assign(pred_dict, target, img_meta)` ile `DynamicTopkAssigner`:
      - LaneIoU + cls/reg cost kombinasyonuna göre anchor–gt eşlemesi yapar.
    - `cls_loss`:
      - Pozitif anchor’lara label=1, diğerlerine 0; Focal loss ile.
    - `reg_xytl_loss`:
      - Anchor parametreleri (`y0, x0, theta, length`) – gerçek değerlerle karşılaştırılır.
    - `iou_loss`:
      - `self.loss_iou` üzerinden LaneIoU cost; x koordinatları gerçek görüntü genişliğine skale edilerek kullanılır.
    - Opsiyonel `loss_seg`:
      - `tgt_masks` CULane segmentation mask’lerinden üretilir.
  - Sonuç: `{"loss_cls", "loss_reg_xytl", "loss_iou", ["loss_seg"]}`.

- `predict`:
  - Son refinement stage’inin çıktılarını alır.
  - NMS + post-processing ile lane koordinatlarını `Lane` nesnelerine veya point-array’lere çevirir.

Özet: Bu head klasik CLRerNet makalesini bire bir yansıtıyor; anchor tabanlı, multi-stage refinement + LaneIoU cost/loss ile confidence kalitesini optimize eden bir yapı.

## CULane Dataset ve Pipeline

### Dataset Sınıfı (`libs/datasets/culane_dataset.py`)

- `CulaneDataset(Dataset)`:
  - `__init__`:
    - `data_root`, `data_list` (örn. `dataset/list/train_gt.txt`).
    - `diff_file` ve `diff_thr`:
      - `train_diffs.npz` içindeki frame farkları ile düşük hareketli/tekrarlı frame’ler filtrelenebiliyor.
    - `self.img_infos`, `self.annotations`, `self.mask_paths`:
      - `parse_datalist` ile dolduruluyor.
    - `self.pipeline = Compose(pipeline)`:
      - MMDetection’in default pipeline’ı yerine kendi transform zinciri kullanılıyor.
  - `parse_datalist`:
    - Her satır: `image_path [mask_path]`.
    - `diffs[i] < diff_thr` ise örnek atlanıyor.
    - Train modunda:
      - `.lines.txt` annotation dosyaları ile lane koordinatları tutuluyor.
  - `prepare_train_img`:
    - OpenCV ile BGR görüntü okunur.
    - `load_labels`:
      - `.lines.txt` okur, `[x0, y0, x1, y1, ...]` şeklinde lane noktalarını çıkarır.
    - `results` dict’i:
      - `img`, `gt_points` (ham lane noktaları), `id_classes`, `id_instances`, `gt_masks` (opsiyonel) ve meta (ori_shape, img_shape).
    - `self.pipeline(results)` çağrılır.
  - `__getitem__`:
    - `test_mode` ise `prepare_test_img`.
    - Train modunda:
      - Pipeline `None` dönerse (örn. augmentasyon filtresi) yeni bir index sample edilir.

### Pipeline Bileşenleri (`libs/datasets/pipelines`)

- `Compose`:
  - Albumentations tabanlı augment’leri (`Alaug`) veya MMDetection transform’larını zincir şeklinde uygular.

- `alaug.py`:
  - Albumentations wrapper’ı; konfig `dict(type="albumentation", pipelines=...)` formatında kullanılır.

- `lane_formatting.py`:
  - `PackCLRNetInputs`:
    - CLRerNet için gerekli lane temsilini (`lanes` tensörü) hazırlar:
      - `sample_lane` ile `offsets_ys` boyunca x koordinatları örneklenir.
      - Her lane için:
        - `y0`, `x0`, `theta`, `length`, `xs` (72 noktalı) formuna dönüştürülür.
    - `DetDataSample` ve `InstanceData` ile MMDetection’in beklediği `data_samples` meta objesini oluşturur.
    - Çıkış: `{"inputs": img_tensor, "data_samples": data_sample}`.
  - `PackLaneLMInputs`:
    - LaneLM başlığı için tamamen ayrı bir temsil üretir:
      - `LaneTokenizerConfig` ile:
        - `img_w=800`, `img_h=320`, `num_steps=40`, `nbins_x=800`.
      - `LaneTokenizer`:
        - Her lane’i (x, y) noktalarından spline ile sürekli fonksiyona çevirir.
        - Y ekseninde `T` adet sabit y konumunda örneklendirir.
        - x’i `[0, nbins_x)` aralığında quantize eder; 0 = padding/no-lane.
        - Y bilgisi, time-step index (`t`) ile temsil edilir; T = padding.
      - `max_lanes` kadar lane için:
        - `lane_tokens_x`: (max_lanes, T)
        - `lane_tokens_y`: (max_lanes, T)
        - `lane_valid_mask`: (max_lanes,) – en az bir non-padding token var mı?
    - `img`:
      - Float’a çevrilip [0, 1] aralığına normalize edilir (CLRerNet training’deki [0–255] + mean/std’den farklı).
    - Çıkış: `{"inputs", "lane_tokens_x", "lane_tokens_y", "lane_valid_mask", "metainfo"}`.

Bu pipeline mimarisi, aynı CULane verisinden iki farklı supervision hattı üretmek için kullanılıyor:
- MMDetection/CLRerNet hattı: anchor + LaneIoU (PackCLRNetInputs).
- LaneLM hattı: token dizileri + dil modeli kaybı (PackLaneLMInputs).

## LaneLM Tarzı Bileşenler

### LaneTokenizer ve Token Temsili (`libs/models/lanelm/tokenizer.py`)

- `LaneTokenizerConfig`:
  - `img_w=800`, `img_h=320`, `num_steps=40`, `nbins_x=800`.
  - `pad_token_x=0`, `pad_token_y=-1` (içsel olarak T’ye map ediliyor).

- `LaneTokenizer`:
  - `T = num_steps`:
    - Her lane, dikey eksende `T` sabit y konumunda örneklenir.
  - `_compute_sample_ys`:
    - [0, img_h) aralığında uniform T nokta.
  - `_fit_spline(points)`:
    - (x, y) noktalarından y’ye göre sıralama.
    - `InterpolatedUnivariateSpline` ile x(y) fonksiyonu.
    - Yine lane geometrisini düzgün hale getirmek için duplicate y’leri temizler.
  - `encode_single_lane`:
    - Girdi: (N, 2) float (x, y) koordinatları.
    - Çıkış:
      - `x_tokens`: (T,) – `[1, nbins_x-1]` aralığında quantize; 0 padding.
      - `y_tokens`: (T,) – `t` veya padding için T.
    - Sınır dışında kalan x’ler için padding devam eder.
  - `decode_single_lane`:
    - Ters dönüşüm: integer token dizilerini tekrar (x, y) sürekli koordinatlarına çevirir.

Bu tasarım, project_analysis.md’de anlatılan LaneLM makalesindeki konseptle bire bir uyumlu:
- Lane = token dizisi.
- Her sample step = y ekseninde sabit bir pozisyon.
- x koordinatı dil modeline “kelime” olarak veriliyor.

### LaneLMModel ve Decoder (`libs/models/lanelm/model.py`)

- `KeypointEmbedding`:
  - `nn.Embedding` tabanlı 3 ayrı embedding:
    - `x_embedding`: `[0, nbins_x)` için.
    - `y_embedding`: `[0, max_y_tokens)` için (time-step/padding).
    - `pos_embedding`: `[0, max_len)` için klasik positional embedding.
  - `forward(x_tokens, y_tokens)`:
    - (B, T, D) şeklinde x + y + pos embedding sum’ı.

- `LaneLMDecoderLayer`:
  - `nn.MultiheadAttention` ile iki aşamalı attention:
    - `self_attn`: causal mask ile token’lar arası otoregresif bağımlılık.
    - `cross_attn`: visual memory üzerinde cross-attention (query=token, key/value=visual).
  - Üç adet `LayerNorm` + FFN bloğu:
    - Transformer decoder standardı.

- `LaneLMDecoder`:
  - `num_layers` adet `LaneLMDecoderLayer`.
  - `_generate_causal_mask(seq_len)`:
    - Üst üçgen boolean mask; geleceğe bakmayı engelliyor.
  - `forward(tgt, memory, memory_key_padding_mask)`:
    - Visual token’lar (memory) sabit; keypoint embedding’leri her layer’da güncelleniyor.

- `LaneLMHead`:
  - İki `Linear`:
    - `proj_x`: embed_dim → nbins_x.
    - `proj_y`: embed_dim → max_y_tokens.
  - Çıkış:
    - `logits_x`: (B, T, nbins_x).
    - `logits_y`: (B, T, max_y_tokens).

- `LaneLMModel`:
  - Parametreler:
    - `nbins_x`, `max_y_tokens`, `embed_dim`, `num_layers`, `num_heads`, `ffn_dim`, `max_seq_len`, `visual_in_dim`.
  - `visual_proj` (opsiyonel):
    - Backbone/FPN’den gelen channel boyutu embed_dim ile eşleşmiyorsa lineer map.
  - `forward(visual_tokens, x_tokens, y_tokens, visual_padding_mask)`:
    - Visual token’ları projelendirir.
    - Keypoint embedding hesaplar.
    - Decoder’e besler.
    - Head ile `logits_x, logits_y` üretir.

Bu yapı, LaneLM makalesinde tarif edilen “visual encoder + Transformer decoder + token prediction head” kombinasyonunun doğrudan bir implementasyonu. 
Buradaki fark, visual encoder’ın (CLRerNet backbone + FPN) tamamen frozen olması ve LaneLM başlığının ayrı bir training script’iyle eğitilmesidir.

## LaneLM Eğitim Script’i (`tools/train_lanelm_culane.py`)

Bu script, MMDetection training loop’undan bağımsız, sade bir PyTorch training loop’u kullanır. Amaç:
- CULane üzerinde CLRerNet’in backbone + FPN’ini sabit görsel encoder olarak kullanmak.
- Her görüntüdeki lane’ler için LaneLM token dizilerini üretmek.
- LaneLMModel’i otoregresif şekilde bu token dizilerini tahmin etmeye zorlamak.

### DataLoader Kurulumu

- `build_culane_lanelm_dataloader(...)`:
  - `configs.clrernet.culane.dataset_culane_clrernet` içinden:
    - `compose_cfg`, `crop_bbox`, `img_scale`, `train_al_pipeline` gibi augmentasyon ayarlarını reuse eder.
  - Yeni `train_pipeline`:
    - `dict(type="albumentation", pipelines=train_al_pipeline)`:
      - Aynı augment’ler; CLRerNet training ile tutarlı.
    - `dict(type="PackLaneLMInputs", ...)`:
      - `max_lanes`, `num_points`, `img_w`, `img_h`, `nbins_x`.
  - `CulaneDataset`:
    - `diff_file` varsa kullanılır; yoksa None.
  - `DataLoader`:
    - `shuffle=True`, `drop_last=True`, `pin_memory=True`.
    - Çıkış batch yapısı:
      - `batch["inputs"]`: (B, 3, H, W) float.
      - `batch["lane_tokens_x"]`: (B, max_lanes, T).
      - `batch["lane_tokens_y"]`: (B, max_lanes, T).
      - `batch["lane_valid_mask"]`: (B, max_lanes).

### Frozen CLRerNet Görsel Encoder

- `build_frozen_clrernet(config_path, checkpoint_path, device)`:
  - `init_detector` (MMDetection) ile CLRerNet + CULane konfig + checkpoint yüklenir.
  - `model.eval()` ve tüm `backbone`, `neck` (ve varsa `bbox_head`) parametreleri `requires_grad=False`.
  - Böylece LaneLM eğitimi sırasında sadece LaneLM başlığı güncellenir.

- `extract_visual_tokens(model, imgs)`:
  - `model.extract_feat(imgs)` ile FPN feature’ları alınır.
  - En yüksek çözünürlüklü seviye `feat = feats[0]` seçilir.
  - Şekil: (B, C, Hf, Wf) → (B, N, C):
    - `feat.view(B, C, Hf * Wf).permute(0, 2, 1)`.
  - `visual_tokens` = patch token’lar; LaneLM decoder’ının memory girişi.

### LaneLM Model Kurulumu

- `build_lanelm_model(...)`:
  - `max_y_tokens = num_points + 1` (padding dahil).
  - `LaneLMModel`:
    - `visual_in_dim` = FPN kanal sayısı (base config’te 64).
    - `embed_dim`, `num_layers`, `num_heads`, `ffn_dim` parametreleri CLI’den geliyor.

### Loss Fonksiyonu

- `lane_lm_loss(logits_x, logits_y, target_x, target_y, pad_token_x, pad_token_y)`:
  - `CrossEntropyLoss(ignore_index=pad_token_x)`:
    - X token’ları için.
  - `CrossEntropyLoss(ignore_index=pad_token_y)`:
    - Y token’ları için (step ID / padding).
  - Toplam loss = 0.5 * (loss_x + loss_y).
  - Target/Logits reshape:
    - (B, T, V) → (B*T, V); (B, T) → (B*T,).

### Eğitim Döngüsü

- `train_one_epoch(...)`:
  - `lanelm_model.train()`.
  - `pad_token_x = 0`, `pad_token_y = num_points`.
  - Her batch:
    - `imgs` → cihaza.
    - `lane_tokens_x`, `lane_tokens_y`, `lane_valid_mask`.
    - Şekil:
      - B = batch size, L = max_lanes, T = `num_points`.
    - `extract_visual_tokens`:
      - (B, N, C_feat).
    - Lanes’i flatten:
      - Visual token’lar: (B, N, C) → (B, L, N, C) → (B*L, N, C).
      - Token dizileri: (B, L, T) → (B*L, T).
    - `valid_mask_flat`:
      - Tamamen padding olan lane’ler atılır.
    - Teacher forcing:
      - Input token’lar = target’ların sağa kaydırılmış versiyonu:
        - İlk token padding, sonra `[t-1]`.
    - LaneLMModel forward:
      - `logits_x, logits_y`.
    - Loss hesaplanır, backprop yapılır, optimizer step.
    - `log_interval` adımda bir loss log’lanır.

- `main()`:
  - CLI argümanları:
    - `--config`, `--checkpoint`, `--data-root`, `--train-list`, `--diff-file`.
    - `--epochs` (default 5), `--batch-size` (default 4), `--num-workers`, `--lr`.
    - LaneLM hyperparametreleri: `--embed-dim`, `--num-layers`, `--num-heads`, `--ffn-dim`, `--num-points`, `--nbins-x`.
    - `--work-dir`.
  - Cihaz seçimi:
    - Eğer CUDA yoksa otomatik CPU’ya düşer.
  - Dataloader → frozen CLRerNet → LaneLMModel → optimizer.
  - Her epoch sonunda:
    - `lanelm_culane_dla34_epoch{epoch}.pth` checkpoint’i kaydedilir.

Bu script, LaneLM başlığının CULane üzerinde ciddi compute olmadan test edilebilmesi için tasarlanmış durumda. Tamamen bağımsız bir loop olduğundan, MMDetection’in default train runner’ına müdahale etmiyor.

## LaneLM Makalesi ve project_analysis.md ile Uyum

`project_analysis.md` LaneLM makalesini ve diğer Transformer tabanlı lane detection çalışmalarını özetliyor. Bu projedeki implementasyonla örtüşen kritik noktalar:

- LaneLM’nin “lane as language” fikri:
  - Her lane, bir token dizisi olarak temsil ediliyor.
  - Bu proje: `LaneTokenizer` + `PackLaneLMInputs` ile bunu bire bir yapıyor.

- Transformer decoder + cross-attention:
  - Makalede, keypoint token’ları query; visual patch’ler key/value.
  - Bu proje: `LaneLMDecoderLayer` aynı yapıyı uyguluyor:
    - `self_attn` (causal) → otoregresif yapı.
    - `cross_attn` (memory=visual_tokens) → görüntü ile etkileşim.

- Visual encoder:
  - Makalede CNN + patch embedding + positional encoding.
  - Bu proje:
    - CLRerNet backbone + FPN zaten lane detection için optimize edilmiş güçlü bir görsel encoder.
    - `extract_visual_tokens` ile patch token’lara dönüştürülüyor.

- Eğitim stratejisi:
  - Makalede genellikle ayrı bir dil modeli başlığı eğitilmesi anlatılır.
  - Bu proje:
    - `build_frozen_clrernet` ile backbone/neck’i tamamen donduruyor.
    - Sadece LaneLM başlığı `AdamW` ile optimize ediliyor.

Farklılıklar:

- Orijinal LaneLM çalışmasının exact hyperparametreleri / loss tanımı burada bire bir kopyalanmamış; bu proje, CLRerNet’in görsel kodlayıcısını kullanarak LaneLM fikrini pratik bir ortamda test etmeye odaklanıyor.
- Görüntü ön-işleme:
  - CLRerNet tarafı mean/std ile normalize ederken, LaneLM pipeline’ında `[0,1]` normalize edilmiş float kullanılıyor; bu, backbone’un pretrain koşulları ile tam hizalı değil, fakat prototip/deneysel amaçlar için kabul edilebilir.

## 1 Epoch, Batch Size 16 Eğitim Senaryosu

Bu projede LaneLM başlığını sadece “çalışıyor mu?” diye görmek için hızlı bir smoke test eğitimini şu şekilde koşturabilirsiniz:

1. CULane dataset sembolik link’i:
   - Repo kökünde `dataset -> /home/alki/projects/CULane` zaten oluşturulmuş olmalı.
   - `dataset/list/train_gt.txt` ve `dataset/list/train_diffs.npz` dosyalarının bu konumda bulunduğunu doğrulayın.

2. CLRerNet pretrained checkpoint’i:
   - Örneğin: `clrernet_culane_dla34_ema.pth` dosyasını repo köküne koyun.

3. 1 epoch, batch size 16 ile LaneLM eğitimi:

```bash
python tools/train_lanelm_culane.py \
  --config configs/clrernet/culane/clrernet_culane_dla34_ema.py \
  --checkpoint clrernet_culane_dla34_ema.pth \
  --data-root dataset \
  --train-list dataset/list/train_gt.txt \
  --diff-file dataset/list/train_diffs.npz \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda \
  --work-dir work_dirs/lanelm_culane_smoketest
```

Bu komut:
- Tüm CLRerNet backbone + FPN’i dondurur.
- CULane train set’inden LaneLM token dizilerini üretir.
- LaneLMModel’i 1 epoch boyunca, batch size 16 ile eğitir.
- Eğitimin sonunda `work_dirs/lanelm_culane_smoketest/lanelm_culane_dla34_epoch1.pth` checkpoint’ini üretir.

## Özet

Bu repo, üç katmanlı bir mimariyi birleştiriyor:
- Alt katmanda, LaneIoU ve anchor bazlı refinement ile SOTA CULane performansı veren CLRerNet.
- Orta katmanda, CULane dataset ve augmentasyon pipeline’ı, hem klasik loss’lar hem de dil-modeli tarzı tokenizasyon için tekrar kullanılıyor.
- Üst katmanda, LaneLM fikrini pratikte test eden, Transformer decoder + token prediction head ve bağımsız bir eğitim script’i.

Tüm bu parçalar bir araya geldiğinde, model hem klasik lane detection metriklerinde güçlü kalırken hem de lane’leri bir dil modeli gibi seri halde üreten yeni bir başlıkla genişletilmiş oluyor.

