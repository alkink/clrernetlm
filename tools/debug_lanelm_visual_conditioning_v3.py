"""
Debug script: test whether LaneLM v3 predictions actually depend on visual
features or collapse to a learned template.

For each image in a (small) list:
  - run autoregressive decoding with normal visual tokens,
  - run autoregressive decoding with zeroed visual tokens,
  - compare the decoded lane coordinates.

If the predictions are almost identical in both cases, the model is not
using visual conditioning effectively.

Usage example (overfit 2-image setup):

PYTHONPATH=/home/alki/projects/clrernetlm:$PYTHONPATH \\
python tools/debug_lanelm_visual_conditioning_v3.py \\
  --config configs/clrernet/culane/clrernet_culane_dla34_ema.py \\
  --clrernet-checkpoint clrernet_culane_dla34_ema.pth \\
  --lanelm-checkpoint work_dirs/lanelm_culane_overfit_2imgs_v3_relative_fix_noaug/lanelm_culane_dla34_v3_epoch200.pth \\
  --data-root dataset \\
  --list-file dataset/list/train_overfit_lanelm.txt \\
  --device cuda
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from mmengine.config import Config

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig, LaneLMModel

from tools.train_lanelm_culane_v3 import (  # type: ignore
    build_frozen_clrernet_backbone,
    extract_pyramid_feats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug LaneLM v3 visual conditioning (normal vs zeroed visual tokens)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="CLRerNet config file.",
    )
    parser.add_argument(
        "--clrernet-checkpoint",
        type=str,
        required=True,
        help="CLRerNet checkpoint.",
    )
    parser.add_argument(
        "--lanelm-checkpoint",
        type=str,
        required=True,
        help="LaneLM v3 checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="CULane dataset root.",
    )
    parser.add_argument(
        "--list-file",
        type=str,
        default="dataset/list/train_overfit_lanelm.txt",
        help="List file with a small number of images.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2,
        help="Number of images to test.",
    )
    parser.add_argument(
        "--max-lanes",
        type=int,
        default=1,
        help="Number of LaneLM lanes to decode per image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use.",
    )
    return parser.parse_args()


def build_culane_dataset_for_debug(data_root: str, list_file: str) -> CulaneDataset:
    """CULane dataset using validation pipeline (Crop+Resize only)."""
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        val_al_pipeline,
    )

    test_pipeline = [
        dict(type="albumentation", pipelines=val_al_pipeline),
    ]

    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_file,
        pipeline=test_pipeline,
        diff_file=None,
        test_mode=True,
    )
    return dataset


def autoregressive_decode_single_lane(
    lanelm_model: LaneLMModel,
    visual_tokens: torch.Tensor,
    num_points: int,
    nbins_x: int,
    max_lanes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode up to max_lanes lanes and return the first lane's tokens."""
    device = visual_tokens.device
    B = visual_tokens.shape[0]
    assert B == 1, "This helper expects batch size 1."
    T = num_points
    pad_token_x = 0
    pad_token_y = T

    all_x = []
    all_y = []

    for lane_idx in range(max_lanes):
        y_tokens = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).clone()
        x_tokens = torch.full(
            (B, T),
            pad_token_x,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(T):
            logits_x, logits_y = lanelm_model(
                visual_tokens=visual_tokens,
                x_tokens=x_tokens,
                y_tokens=y_tokens,
                visual_padding_mask=None,
            )
            step_logits_x = logits_x[:, t, :]
            step_logits_y = logits_y[:, t, :]

            # Greedy decode
            x_next = torch.argmax(step_logits_x, dim=-1)
            y_next = torch.argmax(step_logits_y, dim=-1)

            still_running = ~finished
            x_tokens[still_running, t] = x_next[still_running]
            y_tokens[still_running, t] = y_next[still_running]

            is_eos = (x_next == pad_token_x) | (y_next == pad_token_y)
            finished = finished | is_eos
            if torch.all(finished):
                break

        all_x.append(x_tokens.cpu())
        all_y.append(y_tokens.cpu())

    # Take first lane sequence (shape: (T,))
    x_tokens_all = torch.stack(all_x, dim=1)  # (B, L, T)
    y_tokens_all = torch.stack(all_y, dim=1)
    return x_tokens_all.numpy()[0, 0], y_tokens_all.numpy()[0, 0]


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    cfg = Config.fromfile(args.config)

    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        img_scale,
    )

    img_w, img_h = img_scale

    # Dataset for debug
    dataset = build_culane_dataset_for_debug(args.data_root, args.list_file)
    n = min(args.num_images, len(dataset))
    print(f"Debugging visual conditioning on {n} images.")

    # Frozen CLRerNet
    clrernet_model = build_frozen_clrernet_backbone(
        config_path=args.config,
        checkpoint_path=args.clrernet_checkpoint,
        device=device,
    )

    # LaneLM v3 model
    ckpt = torch.load(args.lanelm_checkpoint, map_location="cpu")
    lm_cfg = ckpt.get("config", {})
    num_points = int(lm_cfg.get("num_points", 40))
    nbins_x = int(lm_cfg.get("nbins_x", 800))
    embed_dim = int(lm_cfg.get("embed_dim", 256))
    num_layers = int(lm_cfg.get("num_layers", 4))
    num_heads = int(lm_cfg.get("num_heads", 8))
    ffn_dim = int(lm_cfg.get("ffn_dim", 512))
    x_mode = lm_cfg.get("x_mode", "relative")
    max_abs_dx = lm_cfg.get("max_abs_dx", 64)
    if max_abs_dx is None:
        max_abs_dx = 64

    visual_in_channels = (64, 64, 64)
    max_y_tokens = num_points + 1
    max_seq_len = num_points

    vocab_size_x = nbins_x
    if x_mode == "relative_disjoint":
        vocab_size_x = nbins_x + 2 * max_abs_dx + 1
        print(f"Disjoint mode: vocab_size_x={vocab_size_x} (abs {nbins_x} + rel {2*max_abs_dx+1})")
    else:
        print(f"Overlapping/absolute mode: vocab_size_x={vocab_size_x}")

    lanelm_model = LaneLMModel(
        nbins_x=vocab_size_x,
        max_y_tokens=max_y_tokens,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        visual_in_dim=None,
        visual_in_channels=visual_in_channels,
    )
    lanelm_model.load_state_dict(ckpt["model_state_dict"])
    lanelm_model.to(device)
    lanelm_model.eval()

    # Tokenizer (we only need it if we want to decode coords; here we compare tokens directly)
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=img_w,
        img_h=img_h,
        num_steps=num_points,
        nbins_x=nbins_x,
        x_mode=x_mode,
        max_abs_dx=max_abs_dx,
    )
    _ = LaneTokenizer(tokenizer_cfg)

    with torch.no_grad():
        for idx in range(n):
            data = dataset[idx]
            img_resized = data["img"]
            img_tensor = (
                torch.from_numpy(img_resized.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            feats = extract_pyramid_feats(clrernet_model, img_tensor)
            visual_tokens = lanelm_model.encode_visual_tokens(feats)  # (1, N, D)

            # Normal decode
            x_norm, y_norm = autoregressive_decode_single_lane(
                lanelm_model=lanelm_model,
                visual_tokens=visual_tokens,
                num_points=num_points,
                nbins_x=nbins_x,
                max_lanes=args.max_lanes,
            )

            # Zeroed visual tokens
            visual_tokens_zero = torch.zeros_like(visual_tokens)
            x_zero, y_zero = autoregressive_decode_single_lane(
                lanelm_model=lanelm_model,
                visual_tokens=visual_tokens_zero,
                num_points=num_points,
                nbins_x=nbins_x,
                max_lanes=args.max_lanes,
            )

            # Compare
            diff_x = np.mean(np.abs(x_norm.astype(np.int32) - x_zero.astype(np.int32)))
            diff_y = np.mean(np.abs(y_norm.astype(np.int32) - y_zero.astype(np.int32)))

            print(f"[Image {idx}] mean |x_norm - x_zero| = {diff_x:.3f}, "
                  f"mean |y_norm - y_zero| = {diff_y:.3f}")
            print(f"  x_norm (first 10): {x_norm[:10]}")
            print(f"  x_zero (first 10): {x_zero[:10]}")


if __name__ == "__main__":
    main()
