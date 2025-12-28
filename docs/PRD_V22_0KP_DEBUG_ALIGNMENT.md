## PRD: V22 — 0‑KP (Prompt’suz) Overfit Debug Hizalaması ve Loss Sadeleştirme

### Amaç
Overfit (özellikle `--overfit-size 1/8`) koşularında modelin **gerçekten 0‑kp** (prompt’suz) çalışmasını sağlamak ve train/test giriş formatı uyuşmazlığını kaldırarak hatanın kaynağını izole etmek.

### Problem Tanımı (Gözlem)
- Test config’inde `use_prompting=False` iken, training tarafında CLRNet/pseudo‑label akışı bazı yerlerde hâlâ devreye giriyor ve/veya pahalı CLRNet yüklemeleri yapılıyor.
- `train_lanelm_v4_fixed.py` içinde Y-loss ağırlıklandırması **convex-combination** biçimindeydi: \((1-w)\cdot L_x + w\cdot L_y\).
  - \(w=1\) olduğunda **X-loss gradient’i sıfırlanabiliyor**; bu da X’in öğrenilmesini bozup şeritlerin şeride oturmamasına katkı verebiliyor.

### Kapsam
Sadece `tools/train_lanelm_v4_fixed.py` üzerinde:
- 0‑kp modunda pseudo-label/prompting için **full CLRNet modelinin yüklenmesini** engellemek
- Loss’ları sadeleştirip overfit debug’u deterministik hale getirmek (varsayılan olarak)
- Y-loss formülünü PDF Eq.11’e uygun hale getirmek (toplamsal)

### Yapılan Değişiklikler
#### 1) 0‑kp modunda CLRNet pseudo-label modelini hiç yükleme
- Yeni argüman: `--num-pseudo-points` (default: `0`)
  - `0` ise: `init_detector(...)` ile full CLRNet (head’li) model **yüklenmiyor** ve batch içinde `predict(...)` çağrısı yapılmıyor.
  - `>0` ise: davranış korunuyor (full CLRNet yükleniyor ve prompt/pseudo-label akışı çalışabiliyor).

#### 2) Overfit debug için loss’ları sadeleştirme (varsayılanlar)
- Yeni argümanlar (varsayılanlar debug odaklı):
  - `--use-y-loss` (default kapalı)
  - `--y-loss-weight` (default `1.0`)
  - `--presence-weight` (default `0.0`)
  - `--ss-max-prob` (default `0.0`)
  - `--ar-rollout-max-weight` / `--ar-rollout-min-weight` (default `0.0`)
  - `--pad-loss-weight` (default `0.0`)
- Hedef: 0‑kp overfit’te önce sadece X davranışını fit etmek; sonra gerekirse loss bileşenlerini tek tek geri eklemek.

#### 3) Y-loss ağırlıklandırma bugfix
- Eski (hatalı/tehlikeli) form: `loss = (1-w)*loss_x + w*loss_y`
- Yeni form (PDF Eq.11’e uygun): `loss = loss_x + (y_loss_weight * loss_y)`

### Kabul Kriterleri
- `--num-pseudo-points 0` ile çalıştırıldığında loglarda **“Loading full CLRerNet model for pseudo labels…”** satırı görülmemeli.
- 0‑kp overfit’te (size 1/8) loss’un hızlı düşmesi ve görsellerde zigzag/hizasızlığın belirgin şekilde azalması beklenir.
- Y-loss açıldığında X-loss’un “öğrenmesi durmamalı” (X gradient’i kapanmamalı).

### Çalıştırma Komutları
0‑kp sade loss (önerilen başlangıç):
```bash
python tools/train_lanelm_v4_fixed.py --overfit-size 8 --epochs 200 --num-pseudo-points 0
```

Y-loss’u sonradan ekleyerek:
```bash
python tools/train_lanelm_v4_fixed.py --overfit-size 8 --epochs 200 --num-pseudo-points 0 --use-y-loss --y-loss-weight 1.0
```

Prompt’lu mod geri denemesi (kontrollü):
```bash
python tools/train_lanelm_v4_fixed.py --overfit-size 8 --epochs 200 --num-pseudo-points 2 --use-y-loss --y-loss-weight 1.0
```






