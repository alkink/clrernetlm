# V14: PDF'ye Göre Düzeltmeler Uygulandı

## Yapılan Kritik Değişiklikler

### 1. Loss Computation Düzeltildi (KRİTİK!)

**Önceki (YANLIŞ):**
```python
# Lgt loss (weight 1.0) + Lq loss (weight 0.5)
loss_x = loss_x_lgt + 0.5 * loss_x_lq
loss_y = loss_y_lgt + 0.5 * loss_y_lq
```

**Yeni (PDF'ye göre DOĞRU):**
```python
# PDF'ye göre: Loss SADECE Lgt kısmında (Lq sadece input, loss yok)
loss_x = loss_x_lgt  # Lq loss'u kaldırıldı!
loss_y = loss_y_lgt  # Lq Y-loss'u kaldırıldı!
```

**PDF Referansı (Sayfa 467-481, Eq. 10, 11):**
- PDF: "Loss. We only adopt standard loss in the decoder-only language models."
- Lq sadece input (query), Lgt output (answer)
- Loss sadece Lgt kısmında hesaplanmalı

### 2. Lq Noise Range Azaltıldı

**Önceki:**
```python
lq_noise_range = 10  # -10 to +10 pixels
```

**Yeni (PDF'ye göre):**
```python
lq_noise_range = 5  # PDF: "randomly shifting the x-coordinates by -5 to 5 pixels" (Section 3.4, line 506)
```

### 3. Batch Size Artırıldı

**Önceki:**
```python
batch_size = 1  # Çok küçük, gradient variance'ı artırır
```

**Yeni:**
```python
batch_size = 8 if args.overfit_size > 1 else 1  # Minimum 8 (PDF'de 128, ama overfit test için 8 yeterli)
```

**PDF Referansı (Sayfa 570):**
- PDF: "128 batch size, 800 nbins and 100 training epochs"
- Batch size=1 çok küçük, gradient variance'ı artırır

### 4. Presence Loss Weight Düşürüldü

**Önceki:**
```python
presence_weight = 0.5  # Çok yüksek
```

**Yeni:**
```python
presence_weight = 0.3  # PDF'de presence head yok, ama GT: 0 lanes için gerekli
```

**Not:** PDF'de presence head bahsedilmiyor, ama GT: 0 lanes durumlarını handle etmek için gerekli. Test için kullanıyoruz.

## PDF vs Bizim Implementasyon - Karşılaştırma

| Özellik | PDF | Bizim (V14 Öncesi) | Bizim (V14 Sonrası) |
|---------|-----|-------------------|---------------------|
| **Loss Computation** | Sadece Lgt | Lgt + Lq (0.5 weight) | ✅ Sadece Lgt |
| **Lq Noise** | -5 to +5 pixels | -10 to +10 pixels | ✅ -5 to +5 pixels |
| **Batch Size** | 128 | 1 | ✅ 8 (overfit test için) |
| **Presence Head** | Yok | Var (weight 0.5) | ✅ Var (weight 0.3) |
| **Y-Loss** | Bahsedilmiyor | Kapalı | ✅ Kapalı |
| **nbins_x** | 800 | 800 | ✅ 800 |
| **Epochs** | 100 | 200+ | ✅ 200+ |
| **Training Strategy** | CLRNet Lq ◦ GT Lgt | CLRNet Lq ◦ GT Lgt | ✅ CLRNet Lq ◦ GT Lgt |

## Beklenen İyileşmeler

1. **Loss Computation Düzeltmesi:**
   - Model artık Lq'yu öğrenmeye çalışmayacak (sadece input olarak kullanacak)
   - "Abrupt change points" problemi azalmalı
   - PDF'deki stratejiye uygun

2. **Lq Noise Azaltılması:**
   - Daha az noise → Daha smooth geçiş Lq → Lgt
   - PDF'deki standard'a uygun

3. **Batch Size Artırılması:**
   - Daha stabil gradient updates
   - Daha iyi training

## Sonraki Adımlar

1. ✅ PDF analizi tamamlandı
2. ✅ Loss computation düzeltildi
3. ✅ Lq noise azaltıldı
4. ✅ Batch size artırıldı
5. ⏳ Modeli yeniden eğit
6. ⏳ Test et ve sonuçları analiz et

## Önemli Notlar

- **Presence Head:** PDF'de yok, ama GT: 0 lanes durumlarını handle etmek için gerekli. Test için kullanıyoruz, training'de de öğrenmesi lazım.
- **Y-Loss:** PDF'de bahsedilmiyor, biz kapalı tutuyoruz (Y token'ları zaten sıralı).
- **AR Loss, Pixel Loss, Pad Loss:** PDF'de bahsedilmiyor, ama training'i stabilize etmek için kullanıyoruz.






