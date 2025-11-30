# LaneLM Overfit Guide (Strict Mode)

## Amaç
Bu doküman, LaneLM modelinin geometrik öğrenme kapasitesini doğrulamak için hazırlanan "strict overfit" prosedürünü açıklar. `train_lanelm_overfit.py` script'i, önceki denemelerdeki veri artırma (augmentation) gürültüsünü ortadan kaldırarak modelin 1-2 görüntüyü ezberleyip ezberleyemediğini test eder.

## Neden Buna İhtiyaç Var?
`LaneLM_DEBUG_JOURNAL.md` analizinde görüldüğü üzere, v3 training sırasında loss değerleri belirli bir seviyenin altına inememiştir (loss_x ~2.5). Bunun temel nedeninin, modelin her epoch'ta farklı augmentasyonlarla (flip, rotation, noise) karşılaşması olduğu düşünülmektedir. Geometriyi öğrenip öğrenemediğini kanıtlamak için varyansı sıfıra indirmemiz gerekir.

## Script: `tools/train_lanelm_overfit.py`

### Özellikler
1. **Clean Pipeline:** Sadece `Crop` ve `Resize` işlemlerini uygular. Flip, Affine, Blur vb. kapalıdır.
2. **Fixed Batch:** Dataset'ten ilk 2 görüntüyü alır ve eğitim boyunca sadece bu batch'i kullanır.
3. **Hybrid Tokenizer:** `x_mode="relative"` ve `max_abs_dx=64` ayarlarını kullanır.
4. **Visualization:** Her 50 epoch'ta bir, modelin o anki haliyle eğitim verisi üzerindeki tahminlerini görselleştirir (`work_dirs/lanelm_overfit_strict/vis/`).

### Kullanım

```bash
python tools/train_lanelm_overfit.py \
  --config configs/clrernet/culane/clrernet_culane_dla34_ema.py \
  --checkpoint clrernet_culane_dla34_ema.pth \
  --data-root dataset \
  --epochs 1000 \
  --lr 1e-3 \
  --device cuda
```

### Beklenen Sonuçlar
- **Loss:** İlk 100-200 epoch içinde loss değerinin hızla düşmesi ve 0'a yaklaşması beklenir (loss < 0.1).
- **Görselleştirme:** `vis/` klasöründeki görsellerde, kırmızı tahmin çizgilerinin (prediction) yeşil/orijinal şeritlerin üzerine tam olarak oturması gerekir.
- **Başarısızlık Durumu:** Eğer loss hala 2.0-3.0 bandında takılıyorsa ve görsellerde şeritler yolun üzerinde değilse (offset problemi), sorun tokenizer veya modelin kapasitesi ile ilgili daha derin bir yerdedir.

## Sonraki Adımlar
Bu test başarılı olursa (loss -> 0):
1. `train_lanelm_culane_v3.py` içinde de augmentation stratejisi gözden geçirilmeli (belki curriculum learning: önce clean, sonra aug).
2. `Teacher Lq` (Noisy Query) mantığı eklenerek modelin genelleme yeteneği artırılmalı.

Test başarısız olursa:
1. `LaneTokenizer`'ın relative modu tekrar incelenmeli (hata birikimi?).
2. `VisualTokenEncoder`'ın FPN feature'larını doğru alıp almadığı kontrol edilmeli.


