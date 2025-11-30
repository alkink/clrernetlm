"""
Debug script to verify if LaneLM is actually attending to visual tokens.

Method:
1. Run inference with real visual tokens -> Get Output 1
2. Run inference with ZERO visual tokens -> Get Output 2
3. Compute difference.

If difference is near zero, the model is ignoring the image (Posterior Collapse).
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import (
    build_frozen_clrernet_backbone,
    extract_pyramid_feats,
    collate_lanelm_batch_v3,
)
from tools.test_lanelm_culane_v2 import autoregressive_decode_tokens_v2
from configs.clrernet.culane.dataset_culane_clrernet import compose_cfg, crop_bbox, img_scale

# Reuse clean pipeline
clean_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Crop", x_min=crop_bbox[0], x_max=crop_bbox[2], y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--clrernet-checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--lanelm-checkpoint", required=True)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-file", default="dataset/list/train_gt.txt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # 1. Load Checkpoint & Model
    print(f"Loading LaneLM checkpoint: {args.lanelm_checkpoint}")
    ckpt = torch.load(args.lanelm_checkpoint, map_location="cpu")
    lm_cfg = ckpt.get("config", {})
    
    # Fallback defaults if config missing in ckpt
    nbins_x = lm_cfg.get("nbins_x", 1024)
    num_points = lm_cfg.get("num_points", 40)
    embed_dim = lm_cfg.get("embed_dim", 256)
    num_layers = lm_cfg.get("num_layers", 4)
    max_lanes = lm_cfg.get("max_lanes", 4)
    
    lanelm = LaneLMModel(
        nbins_x=nbins_x,
        max_y_tokens=num_points + 1,
        embed_dim=embed_dim,
        num_layers=num_layers,
        max_seq_len=40,
        visual_in_channels=(64, 64, 64)
    ).to(device)
    lanelm.load_state_dict(ckpt["model_state_dict"])
    lanelm.eval()

    # 2. Load CLRerNet
    clrernet = build_frozen_clrernet_backbone(args.config, args.clrernet_checkpoint, device)

    # 3. Get One Image
    pipeline = [dict(type="albumentation", pipelines=clean_pipeline)]
    dataset = CulaneDataset(args.data_root, args.list_file, pipeline=pipeline, test_mode=False)
    subset = Subset(dataset, [0]) # First image
    dataloader = DataLoader(subset, batch_size=1, collate_fn=collate_lanelm_batch_v3)
    batch = next(iter(dataloader))
    img = batch["inputs"].to(device)
    
    print("Running Inference with REAL Visual Tokens...")
    feats = extract_pyramid_feats(clrernet, img)
    visual_tokens_real = lanelm.encode_visual_tokens(feats)
    
    x_real, y_real = autoregressive_decode_tokens_v2(
        lanelm, visual_tokens_real, num_points, nbins_x, 1, temperature=0.0
    )
    
    print("Running Inference with ZERO Visual Tokens...")
    visual_tokens_zero = torch.zeros_like(visual_tokens_real)
    
    x_zero, y_zero = autoregressive_decode_tokens_v2(
        lanelm, visual_tokens_zero, num_points, nbins_x, 1, temperature=0.0
    )

    # 4. Compare
    x_real_np = x_real.cpu().numpy().flatten()
    x_zero_np = x_zero.cpu().numpy().flatten()
    
    diff = np.abs(x_real_np - x_zero_np)
    mean_diff = np.mean(diff)
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Real Tokens (First 10): {x_real_np[:10]}")
    print(f"Zero Tokens (First 10): {x_zero_np[:10]}")
    print(f"Mean Abs Difference: {mean_diff:.4f}")
    
    if mean_diff < 0.1:
        print("\nCRITICAL: The model outputs are IDENTICAL.")
        print("Conclusion: The model is IGNORING the visual input (Posterior Collapse).")
        print("It behaves like an unconditional language model just memorizing a mean shape.")
    else:
        print("\nSUCCESS: The model outputs change when visual input is removed.")
        print("Conclusion: The model IS attending to the image.")
        
    print("="*40)

if __name__ == "__main__":
    main()


