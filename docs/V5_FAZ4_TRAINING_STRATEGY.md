## V5 FAZ 4 TAMAMLANDI: Eğitim Stratejisi Güncellemesi

### Scheduled Sampling Güncellemesi
- **Max oran:** %20 → **%50**
- **Başlangıç epoch'u:** 50 → **30**
- **Isınma süresi:** 50 → **40**
- Eğitim, inference'a daha yakın hale getirildi; model kendi tahminlerini daha sık görüyor.

### Autoregressive Rollout Loss
- **Rollout adımı:** 5 step
- **Ağırlık:** 0.3
- Öğretmen zorlamasıyla üretilen diziden başlayıp ilk 5 adımı model tahminleriyle dolduruyoruz.
- Bu adımlar için ek cross-entropy uygulanıyor; kısa AR sekanslarında hata birikimi cezalandırılıyor.

### Padding Bölgesi X=0 Loss (Uzun Vadeli Zigzag Çözümü)
- **Sorun:** GT'nin olmadığı Y aralıklarında (padding timesteps) model eğitilmiyor; inference'ta burada rastgele X üretip zigzag kuyruğu oluşturuyor.
- **Çözüm:** 
  - Y padding (`y_tokens == T`) olan timesteps için ayrı bir X-loss eklendi.
  - Hedef: `x_token = 0` (no-lane) → model bu bölgede X=0 üretmeyi öğreniyor.
  - Uygulama: `loss_x_pad_fn` ile sadece padding mask üzerinde CE, ağırlık ≈ 0.3.
- **Beklenen Etki:**
  - GT'nin olmadığı alt bölgelerde şeritler zorla uzatılmıyor.
  - Decode'da `x_tok == 0` zaman adımları zaten çizilmediği için, kuyruk zigzag'ları uzun vadede kayboluyor.

### Kod Referansı
- `tools/train_lanelm_v4_fixed.py`
  - Satırlar ~300: Scheduled sampling hiperparametreleri
  - Satırlar ~360: `pred_x_tf` her zaman hesaplanıyor
  - Satırlar ~420: AR rollout loss blok
  - Log çıktısı: `AR=` metriği eklendi

### Beklenen Etki
1. **Exposure Bias Azalması:** SS oranı arttığı için model inference koşullarına alışacak.
2. **Zigzag Azalması:** Kısa AR loss, hataların büyümeden düzeltilmesini hedefliyor.
3. **Stabil Eğitim:** SS kademeli arttığı için ani sıçrama yok.

### Sonraki Adım
- **Faz 5:** Inference optimizasyonu (visual-first decode + smoothing güçlendirme)

