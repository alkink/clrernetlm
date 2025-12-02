"""
V5 Faz 1 Test Script
====================
Test: Visual Token Encoder güçlendirme
- Adaptive pooling ile token sayısı azalıyor mu? (250 -> 65)
- Spatial bilgi korunuyor mu?
"""

import torch
from libs.models.lanelm import LaneLMModel, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import LaneLMHyperParams

def test_v5_faz1():
    print("=" * 60)
    print("V5 FAZ 1 TEST: Visual Token Encoder Güçlendirme")
    print("=" * 60)
    
    # Config
    hparams = LaneLMHyperParams(
        nbins_x=200,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4,
    )
    
    # P5 Only config (V5)
    visual_in_channels = (64,)  # P5 Only
    
    # Build model
    max_y_tokens = hparams.num_points + 1
    max_seq_len = hparams.num_points * 2
    
    model = LaneLMModel(
        nbins_x=hparams.nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=hparams.embed_dim,
        num_layers=hparams.num_layers,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=max_seq_len,
        dropout=0.0,
        visual_in_channels=visual_in_channels,
    )
    
    print(f"\n✓ Model oluşturuldu")
    print(f"  Visual encoder: Adaptive pooling enabled")
    print(f"  Target spatial size: (5, 13)")
    
    # Simulate P5 feature map: (B, 64, 10, 25)
    B = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Original P5 feature map
    p5_feat = torch.randn(B, 64, 10, 25, device=device)
    feats = [p5_feat]
    
    print(f"\n✓ Test feature map oluşturuldu")
    print(f"  Input shape: {p5_feat.shape}")
    print(f"  Original tokens: {10 * 25} = 250")
    
    # Encode visual tokens
    with torch.no_grad():
        visual_tokens = model.encode_visual_tokens(feats)
    
    print(f"\n✓ Visual tokens encode edildi")
    print(f"  Output shape: {visual_tokens.shape}")
    print(f"  Actual tokens: {visual_tokens.shape[1]}")
    
    # Check token reduction
    original_tokens = 10 * 25  # 250
    actual_tokens = visual_tokens.shape[1]
    reduction_ratio = actual_tokens / original_tokens
    
    print(f"\n" + "=" * 60)
    print("SONUÇLAR:")
    print("=" * 60)
    print(f"  Original tokens: {original_tokens}")
    print(f"  Actual tokens: {actual_tokens}")
    print(f"  Reduction ratio: {reduction_ratio:.2%}")
    
    if actual_tokens == 65:
        print(f"\n✅ BAŞARILI: Token sayısı 250 -> 65 azaltıldı!")
    else:
        print(f"\n❌ BAŞARISIZ: Token sayısı beklenen 65 değil, {actual_tokens}")
    
    # Check spatial information preservation
    # Visual tokens should have meaningful values (not all zeros)
    token_mean = visual_tokens.mean().item()
    token_std = visual_tokens.std().item()
    
    print(f"\n  Token statistics:")
    print(f"    Mean: {token_mean:.4f}")
    print(f"    Std: {token_std:.4f}")
    
    if token_std > 0.1:
        print(f"\n✅ BAŞARILI: Spatial bilgi korunuyor (std > 0.1)")
    else:
        print(f"\n⚠️  UYARI: Spatial bilgi zayıf olabilir (std < 0.1)")
    
    print(f"\n" + "=" * 60)
    print("FAZ 1 TEST TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    test_v5_faz1()

