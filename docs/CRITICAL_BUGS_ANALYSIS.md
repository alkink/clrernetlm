# KRİTİK BUG'LAR - DETAYLI ANALİZ

## 🔍 BULUNAN SORUNLAR

### **BUG 1: BAŞLANGIÇ TOKEN MİSMATCH** ⭐ EN KRİTİK

**Training (line 359):**
```python
x_in[:, 0] = x_tokens[:, 0]  # GT'nin ilk değeri
```

**Visualization (line 137-144):**
```python
for t in range(T):
    x_in = x_out.clone()
    if t > 0:
        x_in_shifted = torch.zeros_like(x_in)
        x_in_shifted[:, 1:t+1] = x_out[:, :t]
    else:
        x_in_shifted = x_in  # x_in[:, 0] = 0 (PADDING!)
```

**Sorun:**
- Training'de model GT'nin ilk değeri ile başlıyor
- Visualization'da model padding token (0) ile başlıyor
- Model training'de "GT ile başla" öğreniyor, visualize'da "padding ile başla" görüyor
- Bu mismatch zigzag'a neden oluyor!

---

### **BUG 2: AUTOREGRESSIVE DECODE YANLIŞ**

**Visualization (line 137-150):**
```python
for t in range(T):
    x_in = x_out.clone()  # x_out başlangıçta tümü 0
    if t > 0:
        x_in_shifted = torch.zeros_like(x_in)
        x_in_shifted[:, 1:t+1] = x_out[:, :t]  # Önceki token'ları kopyala
        # x_in_shifted[:, 0] = 0 kalıyor! ❌
    else:
        x_in_shifted = x_in  # t=0'da x_in[:, 0] = 0 ❌
    
    logits_x, _ = model(visual_tokens[:1], x_in_shifted, y_fixed, lane_indices=lane_ids)
    pred_x = torch.argmax(logits_x[0, t], dim=-1)
    x_out[0, t] = pred_x
```

**Sorun:**
- Her t'de `x_in_shifted[:, 0] = 0` kalıyor
- Model her t'de "başlangıç token'ı 0" görüyor
- Ama training'de `x_in[:, 0] = x_tokens[:, 0]` (GT'nin ilk değeri)
- Bu mismatch model'i şaşırtıyor!

---

### **BUG 3: Y TOKEN FİLTERİNG EKSİK**

**Visualization (line 163-167):**
```python
valid_mask = x_tokens > 0  # Sadece X token'ları filtreleniyor
x_filtered = x_tokens[valid_mask]
y_filtered = y_tokens[valid_mask]  # Y token'ları da filtreleniyor ama...
```

**Sorun:**
- `y_fixed = torch.arange(T)` (0,1,2,...,T-1) kullanılıyor
- Bu her zaman geçerli (padding yok)
- Ama GT'de bazı Y token'ları padding olabilir (y_tok >= T)
- Decode edilirken `y_tok >= T` kontrolü yapılıyor (line 225)
- Ama visualization'da bu kontrol yapılmıyor!

---

### **BUG 4: DECODE SORUNU**

**Tokenizer (line 225):**
```python
if x_tok == self.cfg.pad_token_x or y_tok >= self.T:
    continue  # Padding token'ları atla
```

**Sorun:**
- Visualization'da `y_fixed = torch.arange(T)` kullanılıyor
- Bu her zaman geçerli (y_tok < T)
- Ama GT'de bazı Y token'ları padding olabilir
- Decode edilirken bu kontrol yapılıyor ama visualization'da y_fixed kullanıldığı için sorun yok
- Ama yine de yanlış!

---

## ✅ ÇÖZÜMLER

### **FIX 1: BAŞLANGIÇ TOKEN DÜZELT**

**Önceki:**
```python
x_in_shifted = x_in  # t=0'da x_in[:, 0] = 0
```

**Yeni:**
```python
# Training'deki gibi: GT'nin ilk değerini kullan
# Ama GT yok, o yüzden model'in kendi tahminini kullan
# VEYA ilk token'ı özel olarak tahmin et
if t == 0:
    # İlk token için özel işlem
    # Model'e boş sequence ver, ilk token'ı tahmin et
    x_in_first = torch.zeros(1, T, dtype=torch.long, device=device)
    logits_x_first, _ = model(visual_tokens[:1], x_in_first, y_fixed, lane_indices=lane_ids)
    pred_x_first = torch.argmax(logits_x_first[0, 0], dim=-1)
    x_out[0, 0] = pred_x_first
    continue  # İlk token'ı atla, sonraki token'lara geç
```

**VEYA:**
```python
# Daha basit: Training'deki gibi başlangıç token'ını kullan
# Ama GT yok, o yüzden model'in kendi tahminini kullan
if t == 0:
    x_in_shifted = torch.zeros(1, T, dtype=torch.long, device=device)
    # İlk token için özel tahmin
    logits_x, _ = model(visual_tokens[:1], x_in_shifted, y_fixed, lane_indices=lane_ids)
    pred_x = torch.argmax(logits_x[0, 0], dim=-1)
    x_out[0, 0] = pred_x
else:
    # Sonraki token'lar için normal autoregressive decode
    x_in_shifted = torch.zeros_like(x_in)
    x_in_shifted[:, 1:t+1] = x_out[:, :t]
    # x_in_shifted[:, 0] = x_out[:, 0]  # İlk token'ı koru!
    logits_x, _ = model(visual_tokens[:1], x_in_shifted, y_fixed, lane_indices=lane_ids)
    pred_x = torch.argmax(logits_x[0, t], dim=-1)
    x_out[0, t] = pred_x
```

---

### **FIX 2: AUTOREGRESSIVE DECODE DÜZELT**

**Önceki:**
```python
x_in_shifted[:, 1:t+1] = x_out[:, :t]
# x_in_shifted[:, 0] = 0 kalıyor!
```

**Yeni:**
```python
x_in_shifted[:, 0] = x_out[:, 0]  # İlk token'ı koru!
x_in_shifted[:, 1:t+1] = x_out[:, :t]
```

---

### **FIX 3: Y TOKEN FİLTERİNG EKLE**

**Önceki:**
```python
valid_mask = x_tokens > 0  # Sadece X token'ları
```

**Yeni:**
```python
valid_mask = (x_tokens > 0) & (y_tokens < T)  # X ve Y token'ları
```

---

## 🎯 ÖNCELİK SIRASI

1. **FIX 1: BAŞLANGIÇ TOKEN DÜZELT** ← EN KRİTİK
2. **FIX 2: AUTOREGRESSIVE DECODE DÜZELT** ← ÖNEMLİ
3. **FIX 3: Y TOKEN FİLTERİNG EKLE** ← İYİLEŞTİRME








