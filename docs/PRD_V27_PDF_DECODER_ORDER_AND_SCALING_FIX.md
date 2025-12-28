## Amaç
LaneLM skorlarının düşük kalması ve “zigzag / yukarı kıvrılma” gibi artefaktların kök nedenlerinden biri olarak tespit edilen **decoder mimari sapmasını** PDF’deki tanıma (Eq.9) geri hizalamak.

## Problem Özeti (Gözlem)
- Test metrikleri düşük; lane’ler yola oturmuyor ve bazı sahnelerde zigzag/kararsızlık görülüyor.
- Kullanıcı analizi: PDF Eq.9’da decoder blok sırası **Self-Attn → Cross-Attn** iken, kodumuzda “V5 Visual-First” olarak **Cross-Attn → Self-Attn** uygulanmış.
- Kodda ayrıca PDF’de geçmeyen ek “heuristic”ler vardı:
  - Self-attn dropout’u `max(dropout, 0.2)` ile **min 0.2**’ye zorlanıyordu.
  - Self-attn çıktısı `* 0.8` ile **scale down** ediliyordu.
  - Cross-attn için `num_heads` bazı durumlarda **iki katına** çıkarılıyordu.
  - Keypoint embedding tarafında `x_embedding_scale` ve `lane_embedding_boost` default’ları “V5” şeklinde agresifti.

## PDF Referansı (Doğrulama)
`pdf_content_dump.txt` içinde:
- Eq.(9) (satır ~387-409):
  - `hi = CrossAtt(Q = CausalSelfAtt(hi−1), K = Li, V = Li)`
  - Bu ifade açıkça **önce causal self-attn**, sonra **cross-attn** sırasını tarif ediyor.

## Root Cause Hipotezi
Decoder blok sırasının ters olması + self-attn’i baskılayan ek heuristikler, AR dizideki “geçmiş token” bilgisini ya yanlış yerden/yanlış şiddette kullanıp ya da görsel koşullamayı bozar. Bu da:
- Lane’lerin sahneye oturmamasına,
- Zigzag/kararsızlık artışına,
- Genelleme skorlarının düşmesine
sebep olabilir.

## Uygulanan Değişiklikler
### 1) Decoder attention sırası PDF’e hizalandı
Dosya: `libs/models/lanelm/model.py`
- `LaneLMDecoderLayer.forward` akışı:
  - **Önce** causal **self-attention**
  - **Sonra** **cross-attention (visual tokens)**
  - FFN aynı şekilde devam
- Aşağıdaki V5 sapmaları kaldırıldı:
  - self-attn `dropout=max(dropout, 0.2)` → `dropout=dropout`
  - self-attn output `*0.8` ölçekleme kaldırıldı
  - cross-attn head sayısını ikiye katlama kaldırıldı (`num_heads` korunuyor)

### 2) Keypoint embedding scaling/boost default’ları nötrleştirildi
Dosya: `libs/models/lanelm/model.py`
- `KeypointEmbedding`:
  - `x_embedding_scale` default **1.0**
  - `lane_embedding_boost` default **1.0**
- `LaneLMModel`:
  - `x_embedding_scale` default **1.0**
  - `lane_embedding_boost` default **1.0**

### 3) Train script default’ları güncellendi
Dosya: `tools/train_lanelm_v4_fixed.py`
- `build_lanelm_model_v4(...)` default’ları `x_embedding_scale=1.0`, `lane_embedding_boost=1.0`
- CLI arg `--lane-embedding-boost` default’u **1.0** yapıldı (paper’a yakın)

## Beklenen Etki / Risk
- **Beklenen**: daha stabil decoding, daha iyi genelleme (özellikle “zigzag” ve “lane oturmama” problemlerine katkı).
- **Risk**: Bu değişiklikler mimariyi değiştirdiği için **mevcut checkpoint’ler artık uyumlu değil**; yeniden eğitim gerekir.

## Doğrulama Planı
- 1-image overfit (kısa smoke): loss düşüyor mu, görsel olarak lane oturuyor mu?
- train_100 → test_100: metrik + görsel çıktılar (presence_filter politikası net: presence loss yoksa filter kapalı).

## Roadmap Notu
- `docs/roadmap.md` bu repoda bulunamadı; ileride eklenirse bu PRD ilgili hedeflere bağlanmalı.


