"""
Cross-Attention Debug Script.

Purpose:
  Analyze if the LaneLM decoder is actually using visual information.
  Visualize cross-attention weights to see where the model "looks" in the image.

Diagnosis:
  - If attention is uniform/random → Visual encoder problem
  - If attention focuses on wrong areas → Decoder training problem
  - If attention focuses correctly but wrong output → Head/tokenizer problem
"""

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig

from tools.train_lanelm_culane_v3 import (
    LaneLMHyperParams,
    build_frozen_clrernet_backbone,
    build_lanelm_model_v3,
)

from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg,
    crop_bbox,
    img_scale,
)


class LaneLMModelWithAttention(nn.Module):
    """Wrapper to extract cross-attention weights."""
    
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.cross_attn_weights = []
        
        # Hook into decoder layers to capture cross-attention
        for layer in self.model.decoder.layers:
            layer.cross_attn.register_forward_hook(self._hook_cross_attn)
    
    def _hook_cross_attn(self, module, input, output):
        # output is (attn_output, attn_weights) when need_weights=True
        # But our model uses need_weights=False, so we need to modify
        pass
    
    def forward_with_attention(self, visual_tokens, x_tokens, y_tokens, lane_indices=None):
        """Forward pass that also returns cross-attention weights."""
        self.cross_attn_weights = []
        
        # Manually run through the model to capture attention
        # Embed keypoints (correct attribute name)
        keypoint_emb = self.model.keypoint_embed(x_tokens, y_tokens, lane_indices)
        
        # Visual tokens are already encoded
        memory = visual_tokens
        
        # Run decoder with attention capture
        tgt = keypoint_emb
        causal_mask = self.model.decoder._generate_causal_mask(tgt.shape[1], tgt.device)
        
        all_attn_weights = []
        for layer in self.model.decoder.layers:
            # Self-attention
            residual = tgt
            attn_out, _ = layer.self_attn(tgt, tgt, tgt, attn_mask=causal_mask, need_weights=False)
            tgt = layer.norm1(residual + layer.dropout(attn_out))
            
            # Cross-attention WITH weights
            residual = tgt
            attn_out, attn_weights = layer.cross_attn(
                tgt, memory, memory, 
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=True  # Average over heads
            )
            all_attn_weights.append(attn_weights)
            tgt = layer.norm2(residual + layer.dropout(attn_out))
            
            # FFN
            residual = tgt
            ffn_out = layer.ffn(tgt)
            tgt = layer.norm3(residual + layer.dropout(ffn_out))
        
        # Head
        logits_x = self.model.head.proj_x(tgt)
        logits_y = self.model.head.proj_y(tgt)
        
        return logits_x, logits_y, all_attn_weights


def extract_p5_feat(model, imgs):
    with torch.no_grad():
        feats = model.extract_feat(imgs)
        p5 = feats[-1]  # Highest level, lowest resolution
    return [p5]


def visualize_attention(img, attn_weights, feat_h, feat_w, save_path, title="Cross-Attention"):
    """Visualize cross-attention weights as heatmap overlay."""
    # attn_weights: (T, N) where N = feat_h * feat_w
    # Average over all query positions (T)
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()  # (N,)
    
    # Reshape to spatial dimensions
    attn_map = avg_attn.reshape(feat_h, feat_w)
    
    # Resize to image size
    attn_resized = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
    
    # Normalize for visualization
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay on image
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    
    # Save
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attn_map, cmap='jet')
    plt.colorbar()
    plt.title(f"Attention Map ({feat_h}x{feat_w})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Attention Overlay")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_attention_distribution(attn_weights, name=""):
    """Analyze if attention is uniform or focused."""
    # attn_weights: (T, N)
    attn_np = attn_weights.cpu().numpy()
    
    # Entropy of attention distribution (higher = more uniform)
    entropy_per_query = []
    for t in range(attn_np.shape[0]):
        p = attn_np[t] + 1e-10  # Add small epsilon
        p = p / p.sum()  # Normalize
        entropy = -np.sum(p * np.log(p))
        entropy_per_query.append(entropy)
    
    avg_entropy = np.mean(entropy_per_query)
    max_possible_entropy = np.log(attn_np.shape[1])  # log(N)
    uniformity = avg_entropy / max_possible_entropy  # 0 = focused, 1 = uniform
    
    # Peak attention value
    max_attn = attn_np.max()
    min_attn = attn_np.min()
    
    print(f"{name} Attention Analysis:")
    print(f"  Shape: {attn_np.shape}")
    print(f"  Max attention: {max_attn:.4f}")
    print(f"  Min attention: {min_attn:.6f}")
    print(f"  Avg entropy: {avg_entropy:.4f} (max possible: {max_possible_entropy:.4f})")
    print(f"  Uniformity score: {uniformity:.4f} (0=focused, 1=uniform)")
    
    if uniformity > 0.9:
        print(f"  ⚠️ WARNING: Attention is very UNIFORM - model may be ignoring visual info!")
    elif uniformity < 0.5:
        print(f"  ✅ Attention is FOCUSED - model is looking at specific areas")
    else:
        print(f"  ⚡ Attention is PARTIALLY focused")
    
    return uniformity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--backbone-checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--checkpoint", default="work_dirs/lanelm_2k_subset/lanelm_2k_best.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--work-dir", default="work_dirs/cross_attn_debug")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Model config (must match training)
    nbins_x = 200
    max_abs_dx = 32
    total_vocab_size = 300
    bos_token_ids = [296, 297, 298, 299]

    hparams = LaneLMHyperParams(
        nbins_x=total_vocab_size,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4,
    )
    
    # Load models
    print("Loading CLRerNet backbone...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.backbone_checkpoint, device)
    
    print("Loading LaneLM...")
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        lanelm.load_state_dict(ckpt["model_state_dict"])
    else:
        lanelm.load_state_dict(ckpt)
    lanelm.eval()
    
    # Wrap model to get attention weights
    lanelm_attn = LaneLMModelWithAttention(lanelm).to(device)
    lanelm_attn.eval()
    
    # Tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w, img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=nbins_x,
        x_mode="relative_disjoint",
        max_abs_dx=max_abs_dx
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Test images with GT
    test_images = [
        "driver_100_30frame/05251517_0433.MP4/04500.jpg",
        "driver_100_30frame/05251517_0433.MP4/04530.jpg",
        "driver_100_30frame/05251517_0433.MP4/04560.jpg",
    ]
    
    # Clean pipeline
    test_pipeline = [
        dict(type="Compose", params=compose_cfg),
        dict(type="Crop", x_min=crop_bbox[0], x_max=crop_bbox[2],
             y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
        dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    ]
    
    print("\n" + "="*60)
    print("CROSS-ATTENTION ANALYSIS")
    print("="*60)
    
    uniformity_scores = []
    
    for idx, img_rel_path in enumerate(test_images[:args.num_samples]):
        img_path = os.path.join(args.data_root, img_rel_path)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Load and preprocess image
        img_orig = cv2.imread(img_path)
        
        # Apply crop and resize manually
        crop_y, crop_x = crop_bbox[1], crop_bbox[0]
        crop_h = crop_bbox[3] - crop_bbox[1]
        crop_w = crop_bbox[2] - crop_bbox[0]
        img_cropped = img_orig[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        img_resized = cv2.resize(img_cropped, (img_scale[0], img_scale[1]))
        
        # Prepare tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Extract visual features
            feats = extract_p5_feat(clrernet, img_tensor)
            visual_tokens = lanelm.encode_visual_tokens(feats)
            
            # Get feature map dimensions
            _, _, feat_h, feat_w = feats[0].shape
            
            print(f"\n--- Image {idx+1}: {img_rel_path} ---")
            print(f"Feature map size: {feat_h}x{feat_w} = {feat_h*feat_w} tokens")
            
            # For each lane, analyze attention
            for lane_idx in range(4):
                # Create input tokens (BOS + padding)
                T = tokenizer.cfg.num_steps
                x_in = torch.zeros(1, T, dtype=torch.long, device=device)
                x_in[0, 0] = bos_token_ids[lane_idx]
                y_in = torch.zeros(1, T, dtype=torch.long, device=device)
                lane_indices = torch.tensor([lane_idx], dtype=torch.long, device=device)
                
                # Forward with attention
                logits_x, logits_y, attn_weights_list = lanelm_attn.forward_with_attention(
                    visual_tokens, x_in, y_in, lane_indices
                )
                
                # Analyze last layer attention
                last_layer_attn = attn_weights_list[-1][0]  # (T, N)
                uniformity = analyze_attention_distribution(
                    last_layer_attn, 
                    f"Lane {lane_idx}"
                )
                uniformity_scores.append(uniformity)
                
                # Visualize attention for first lane only
                if lane_idx == 0:
                    save_path = os.path.join(args.work_dir, f"attn_img{idx}_lane{lane_idx}.png")
                    visualize_attention(
                        img_resized, 
                        last_layer_attn,
                        feat_h, feat_w,
                        save_path,
                        title=f"Image {idx+1}, Lane {lane_idx} Cross-Attention"
                    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    avg_uniformity = np.mean(uniformity_scores)
    print(f"Average Uniformity Score: {avg_uniformity:.4f}")
    
    if avg_uniformity > 0.85:
        print("\n🚨 DIAGNOSIS: SEVERE POSTERIOR COLLAPSE")
        print("   Model is NOT using visual information!")
        print("   Cross-attention is nearly uniform across all visual tokens.")
        print("\n   RECOMMENDED FIXES:")
        print("   1. Use Full FPN (P3+P4+P5) instead of P5-only")
        print("   2. Add 2D positional embedding to visual tokens")
        print("   3. Reduce visual token count with pooling")
        print("   4. Increase cross-attention temperature/sharpness")
    elif avg_uniformity > 0.7:
        print("\n⚠️ DIAGNOSIS: WEAK VISUAL ATTENTION")
        print("   Model is partially using visual information but not focused.")
        print("\n   RECOMMENDED FIXES:")
        print("   1. Train longer with curriculum learning")
        print("   2. Add attention regularization loss")
    else:
        print("\n✅ DIAGNOSIS: VISUAL ATTENTION WORKING")
        print("   Model is focusing on specific visual regions.")
        print("   Problem may be in tokenization or decoder capacity.")
    
    print(f"\nVisualizations saved to {args.work_dir}/")


if __name__ == "__main__":
    main()

