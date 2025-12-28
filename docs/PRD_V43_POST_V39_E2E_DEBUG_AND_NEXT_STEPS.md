## PRD_V43 — V39 Sonrası E2E Debug Timeline + Next Steps

Bu doküman, `docs/PRD_V39_FULL2K_FAIL_DEBUG.md` sonrasındaki tüm önemli değişiklikleri, koşuları, metrikleri ve “kanıta dayalı” bulguları tek yerde toplar.

---

### 1) V39 → V40: Full-2k ve AR stabilizasyon denemeleri

#### V39 (referans)
`PRD_V39_FULL2K_FAIL_DEBUG.md`:
- Full-2k (overfit=0) test_100: F1@0.5 ≈ **0.07**
- HR ON/OFF fark etmedi
- Train log: TF iyi, AR kötü → **exposure bias/AR drift**

#### V40: AR-stable ayarlar
Amaç: AR drift’i azaltmak için scheduled sampling + AR rollout loss + pad-loss.

Kullanılan tipik ayarlar:
- `--ss-max-prob 0.2`
- `--ar-rollout-max-weight 0.05 --ar-rollout-min-weight 0.02`
- `--pad-loss-weight 1.0`

**Bulgular:**
- overfit-size sweep (1/8/32/64) yapıldı.
- Test tarafında bazı koşularda lane sayısı her dosyada 4’e “kilitlenmiş” göründü (presence/slot filtering yokken beklenen).

Örnek metrik (V40 overfit32 arstable):
- `work_dirs/v40_test100_overfit32_arstable/20251226_064354`: **F1@0.5 = 0.1746**

---

### 2) V40: “Train vis iyi ama test F1 kötü” farkının analizi

#### Kanıt 1 — Train görselleştirme ≠ Eval metriği
- Train `vis` görüntüleri, training set’ten alınan sabit batch üzerinde çiziliyor.
- Train `vis` çizimi `smooth=True` ile yapılıyor (metric’te `smooth=False`).
- CULane metric pipeline: `.lines.txt` yaz → `Lane(points)` spline ile yeniden örnekle → IoU hesapla.

Bu yüzden “train overlay güzel” tek başına F1@0.5 garantisi değil.

#### Kanıt 2 — Presence filter yokken 4 lane üretimi
Prediction dosyaları istatistiği (V40):
- overfit1/8/32: **100/100 dosyada lane_count=4**
- overfit64: 5/100 dosya boş, kalanların çoğu lane_count=4

Bu durum precision’ı düşürür; F1@0.5’i baskılar.

---

### 3) V41: Presence eğitim/filtre ablation
Hipotez: presence başı eğitilirse ve filtre açılırsa FP azalır.

Sonuç: pratikte büyük bir sıçrama görülmedi (bu aşamada asıl bottleneck strict IoU / AR stabilite).

---

### 4) V42: PDF’e hizalı prompting denemeleri (CLRNet Lq)

#### V42-0: Prompt2 ilk deneme (çöküş)
`work_dirs/v42_test100_overfit32_prompt2`:
- F1@0.5 ≈ **0.018** civarına düştü.

**Bulgular:**
- Prompt açılınca `empty_files` arttı ve `lanes_mean` düştü (lane drop / FN artışı).

#### V42-1: Fixed-Y prompting düzeltmeleri
Amaç: “Lq ◦ Lgt” yaklaşımını fixed-y tokenizasyonla uyumlu hale getirmek.

Yapılan değişiklikler:
- Inference: prompt’u prefix gibi değil, **T=40 sparse x token** olarak doğru `t` indekslerine yerleştir.
- Training: tokenları başa “sıkıştırma” yok; prompt `t`’leri loss’tan çıkar (Lq mask).
- Ek: CLRNet lane slot sırası için inference’ta soldan-sağa sort eklendi.

#### V42 sonuçları (mevcut kanıtlar)
`work_dirs/v42_test100_overfit32_prompt2_fix/20251226_161150`:
- F1@0.5 = **0.0707**

`work_dirs/v42_test100_overfit32_prompt2_fix_sorted/20251226_164146`:
- F1@0.5 = **0.0764**

Bu değerler V40 overfit32 baseline (**0.1746**) altında kalıyor → **prompting hâlâ net fayda sağlamıyor**.

---

### 5) Kritik Meta-Bulgu: Prompt “neden yardımcı olmuyor?” (kanıt odaklı next step)

Şu anki en kuvvetli (test edilebilir) hipotez:
- Bizde seq yönü `t=0` → **y=top** (tokenizer sample_ys: 0→H).
- Causal self-attn sebebiyle, bir prompt token’ın etkisi **ancak daha sonraki t’lere** akar.
- Eğer CLRNet’in “ilk 2 keypoint”i gerçekte lane’in **altına (ego’ya yakın)** karşılık geliyorsa,
  bu noktalar **büyük y** ve dolayısıyla **büyük t** olur → prompt çok geç gelir ve decode’u “yönlendiremez”.

Bu, paper’daki “iki adjacent keypoint” prompting’in neden çalışmadığını açıklayabilir.

**Bunu kanıtlamak için:**
- mmengine log’a `prompt_t` indeks dağılımını yazdır (ilk batch/sample).
- Eğer prompt_t çoğunlukla yüksekse → çözüm aday: **sequence yönünü bottom→top** çevirmek.

---

### 6) Next Actions (V44 plan)
1) **Prompt_t dağılımını logla** (MMLogger ile; print kaybolmasın).
2) Prompt_t yüksek çıkarsa:
   - `LaneTokenizer._compute_sample_ys()` yönünü bottom→top yap
   - encode/decode + y_fixed semantiğini buna göre hizala
   - küçük bir overfit32 + prompt2 koşusu ile tekrar test et

---

### 7) Kanıt: `_debug_prompt_t_indices` (2025-12-26)

Run:
- `work_dirs/_debug_prompt_t_indices/20251226_232329`

Metrik:
- F1@0.5 = **0.0315** (prompting bu koşuda belirgin şekilde zararlı)

Log bulgusu (ilk batch / lane0):
- CLRNet ilk lane noktaları **normalize** görünüyor:
  - `Y range: [0.9084, 1.0000]` (yani görüntünün altına çok yakın)
- Bu noktalar fixed-y grid’e map edilince:
  - `prompt_t_indices=[33, 34]`

Yorum:
- Mevcut `top_to_bottom` diziliminde `t=0` üst (y=0) olduğu için,
  lane’in altından gelen prompt’lar **çok geç** (t≈33/34) enjekte ediliyor.
- Causal self-attn yüzünden bu prompt’lar **erken timestep’leri etkileyemez**;
  prompting’in “refine” etkisi beklenen şekilde çalışmıyor olabilir.


