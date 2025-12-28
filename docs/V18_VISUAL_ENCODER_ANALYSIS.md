# V18: Visual Encoder Analizi - P3+P4+P5 vs P5-Only

## Test Sonuçları (20251207_182544, 20251207_183521)

- **F1@0.5: ~0.02** (hala çok kötü)
- **"Şeritler yukarı yükseliyor"**: Devam ediyor
- **Şeritlere oturmuyor**: Devam ediyor

### Kullanıcının Yeni Bulgusu:

**"Visual encoder: PDF'de P3+P4+P5, bizde P5-only"**

Bu çok kritik bir bulgu! Visual encoder'ın tam olarak nasıl kullanıldığını kontrol etmeliyiz.

## PDF'de Visual Encoder

### PDF (Line 344-365):

**Visual encoder. Our visual encoder has to extract visual features from images and then transform
them into embedding sequences. The pyramid feature extractor f shown in Figure 3 adopts the
standard ResNet[38] and DLA[39] as visual feature extractor. And we add a FPN [40] neck to provide
integrated multi-scale features:**

```
{F0, F1, F2} = f (Xv)
```

**where Ci, Hi and Wi is the channel, the height and the width of the i-th level feature Fi ∈ RB×Ci×Hi×Wi
extracted from CNN and FPN [40].This structure leverages ConvNet's pyramidal feature hierarchy
and demonstrates its efficiency in [6,13,16,41]. Then we split Fi into fixed-size patches, linearly embed
each of them, add position embeddings:**

```
Li = Ev(Fi) + PEvision(Hi, Wi) + LE(i)
```

**where LE(·) is level embedding that embeds level information into vectors, Ev(·) is patch embedding
in ViT[42], PEvision(·) is its standard positional encoding layer that retains the positional information
of each patches and the result value Li ∈ RB×Ni×D represents the token sequence extracted from level
feature Fi, in which Ni is the number of patches and we linearly embed them into D-dimensional
visual embeddings to aligned with keypoint embeddings e ∈ RD.**

PDF'de **{F0, F1, F2}** kullanılıyor, yani **3 seviye FPN** (P3, P4, P5)!

### PDF Ablation Study (Table 5, Line 1512-1612):

**"FPN" kolonu var!** Bu, FPN kullanımının ablation study'de test edildiğini gösteriyor.

**Baseline:** LaneLM-512 (0-kp) with DLA34 encoder but **without HR** (Hallucination Removal)

**FPN kolonu:**
- FPN yok: F1 = 68.36 (-2.35)
- FPN var: F1 = 70.71 (baseline)
- FPN + LE: F1 = 75.54 (+4.83)
- FPN + LE + HR: F1 = 77.80 (+2.26)
- FPN + LE + HR + 2-kp: F1 = 79.04 (+1.24)
- FPN + LE + HR + 2-kp + cmd: F1 = 82.71 (+3.67)

**Sonuç:** PDF'de **FPN kullanılıyor** ve **önemli bir performans artışı sağlıyor** (+2.35 F1)!

## Bizim Implementasyon

### Kod (train_lanelm_v4_fixed.py, line 220-226):

```python
# 2. P5 ONLY (reduced noise, ~250 tokens instead of ~6,500)
# Full FPN: (64, 64, 64) → P5 Only: (64,)
use_p5_only = True  # FIX 1: Reduce visual token noise
if use_p5_only:
    visual_in_channels = (64,)  # P5 Only
else:
    visual_in_channels = (64, 64, 64)  # Full FPN
```

**Sorun:** `use_p5_only = True` kullanıyoruz, ama PDF'de **Full FPN (P3+P4+P5)** kullanılıyor!

## PDF vs Bizim - Visual Encoder

| Özellik | PDF | Bizim (V18 Öncesi) | Durum |
|---------|-----|-------------------|-------|
| **FPN Levels** | P3+P4+P5 (3 levels) | P5 Only (1 level) | ❌ **YANLIŞ!** |
| **visual_in_channels** | (64, 64, 64) | (64,) | ❌ **YANLIŞ!** |
| **Level Embedding (LE)** | Var | ? | ? |
| **Visual Tokens** | ~6,500 (P3+P4+P5) | ~250 (P5 only) | ❌ **ÇOK AZ!** |

## Neden Bu Önemli?

1. **Visual Information:** P5-only sadece en yüksek seviye (en abstract) feature'ları kullanıyor. P3+P4+P5 ise multi-scale feature'ları kullanıyor, bu da daha zengin spatial bilgi sağlıyor.

2. **PDF Ablation Study:** FPN kullanımı +2.35 F1 artışı sağlıyor! Bu çok önemli!

3. **"Şeritler yukarı yükseliyor" Sorunu:** Visual encoder'ın yetersiz olması, modelin y koordinatlarını doğru öğrenememesine neden olabilir.

## Çözüm

### 1. Full FPN'e Geç (P3+P4+P5)

```python
use_p5_only = False  # V18: PDF'de Full FPN kullanılıyor!
visual_in_channels = (64, 64, 64)  # Full FPN
```

### 2. Level Embedding (LE) Kontrol Et

PDF'de Level Embedding (LE) var. Bizim kodda var mı kontrol etmeliyiz.

### 3. Visual Token Count

Full FPN ile visual token sayısı artacak (~6,500), bu da daha fazla bilgi demek.

## Sonraki Adımlar

1. **Full FPN'e Geç:** `use_p5_only = False`, `visual_in_channels = (64, 64, 64)`
2. **Level Embedding Kontrol Et:** PDF'de var, bizde var mı?
3. **Modeli Yeniden Eğit:** Full FPN ile
4. **Test Et:** "Şeritler yukarı yükseliyor" sorunu düzelmiş mi?






