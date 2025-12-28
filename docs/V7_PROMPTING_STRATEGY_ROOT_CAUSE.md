# V7: Prompting Strategy Root Cause Analysis

## Problem
Model prompting strategy'yi kullanamıyor. Test'te CLRNet'ten gelen initial keypoint'ler (`x_tok_2=[531, 534]`) doğru tokenize ediliyor, ama model t=2'de padding token (0) üretiyor (%99.98 olasılıkla).

## Root Cause
**Training'de pseudo label kullanılmıyor!** PDF'deki Equation 10'a göre:
```
S = (L1_q ◦ L1_gt, ..., LN_q ◦ LN_gt, Xv)
```
- `Lq`: CLRNet'ten gelen pseudo keypoint labels (query)
- `Lgt`: Ground truth keypoint labels (answer)
- `◦`: Concatenation

Ama bizim `train_lanelm_v4_fixed.py` kodumuzda:
- Direkt GT kullanılıyor (`x_tokens` = GT tokens)
- Pseudo label (`Lq`) hiç kullanılmıyor
- Model `Lq → Lgt` mapping'ini öğrenmiyor

## Evidence
1. **Training kodunda pseudo label yok:**
   - `train_lanelm_v4_fixed.py` line 374: `x_np, y_np = tokenizer.encode_single_lane(pts)` - Direkt GT encode ediliyor
   - CLRNet'ten pseudo label alınmıyor
   - `Lq ◦ Lgt` formatı yok

2. **Test'te model padding token üretiyor:**
   - `[DEBUG] Lane 0, t=2: pred_x=0, logits_range=[-6.96, 15.12], mean=-2.05, top5_tokens=[0, 7, 295, 351, 281], top5_probs=[0.9997978806495667, ...]`
   - Model %99.98 olasılıkla padding token (0) tahmin ediyor
   - İlk 2 keypoint (`x_tok_2=[531, 534]`) doğru tokenize ediliyor ama model bunları kullanamıyor

3. **PDF'deki training stratejisi:**
   - Section 3.4: "Self-supervised labels" - CLRNet pseudo labels kullanılıyor
   - Section 3.4: "Prompting strategy (1)" - CLRNet'ten 2 initial keypoint alınıyor
   - Model `Lq → Lgt` mapping'ini öğreniyor, bu yüzden inference'da CLRNet keypoint'lerini kullanabiliyor

## Solution
Training'e pseudo label eklemeliyiz:

1. **CLRNet'ten pseudo labels al:**
   - Her image için CLRNet predict et
   - Pseudo keypoint labels (`Lq`) tokenize et
   - GT keypoint labels (`Lgt`) tokenize et

2. **Training sequence oluştur:**
   - `Lq ◦ Lgt` formatında concatenate et
   - Model `Lq`'yu görüp `Lgt`'yi tahmin etmeye çalışsın

3. **Loss hesapla:**
   - `Lq` kısmı için loss yok (sadece input)
   - `Lgt` kısmı için loss var (target)

## Implementation Plan
1. Training loop'a CLRNet pseudo label extraction ekle
2. `Lq ◦ Lgt` formatında sequence oluştur
3. Model'in `Lq` kısmını görmesini, `Lgt` kısmını tahmin etmesini sağla
4. Loss'u sadece `Lgt` kısmı için hesapla

## Expected Impact
- Model prompting strategy'yi öğrenecek
- Test'te CLRNet'ten gelen initial keypoint'leri kullanabilecek
- F1@0.5 skoru artacak (şu an 0.0165, hedef: >0.5)








