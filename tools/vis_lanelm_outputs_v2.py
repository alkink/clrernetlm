"""
Visualize LaneLM v2 predictions vs ground-truth lanes on a small subset of CULane.

This script:
  - Loads CLRerNet backbone + FPN (frozen) and LaneLM v2 checkpoint.
  - Runs LaneLM v2 inference on a few test images.
  - Draws GT lanes and predicted lanes on the original images and saves PNGs.

Usage (inside conda 'clrernet' env):

PYTHONPATH=/home/alki/projects/clrernetlm:$PYTHONPATH \\
python tools/vis_lanelm_outputs_v2.py \\
  --config configs/clrernet/culane/clrernet_culane_dla34_ema.py \\
  --clrernet-checkpoint clrernet_culane_dla34_ema.pth \\
  --lanelm-checkpoint work_dirs/lanelm_culane_dla34_v2_full/lanelm_culane_dla34_v2_epoch3.pth \\
  --data-root dataset \\
  --test-list dataset/list/test.txt \\
  --num-images 20 \\
  --out-dir work_dirs/lanelm_culane_dla34_v2_full/vis
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.config import Config

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig, LaneLMModel
from libs.utils.lane_utils import Lane

from tools.train_lanelm_culane_v2 import (  # type: ignore
    build_frozen_clrernet_backbone,
    extract_pyramid_feats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LaneLM v2 predictions vs GT on CULane."
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
        help="LaneLM v2 checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="CULane dataset root.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="dataset/list/test.txt",
        help="CULane test list file.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of images to visualize.",
    )
    parser.add_argument(
        "--max-lanes",
        type=int,
        default=4,
        help="Max number of lanes to decode per image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save visualization PNGs.",
    )
    return parser.parse_args()


def build_culane_test_dataset_for_vis(
    data_root: str,
    list_path: str,
) -> CulaneDataset:
    """CULane dataset with albumentations for images.

    We set test_mode=True so that CulaneDataset does not try to read
    annotations internally; GT lanes are loaded separately from the
    corresponding .lines.txt files using the relative image path.
    """
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        val_al_pipeline,
    )

    test_pipeline = [
        dict(type="albumentation", pipelines=val_al_pipeline),
    ]

    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=test_pipeline,
        diff_file=None,
        test_mode=True,
    )
    return dataset


def load_gt_lanes_for_image(
    data_root: str,
    img_rel_path: str,
) -> List[np.ndarray]:
    """Load GT lanes from .lines.txt for a single image in original coord system.

    We reconstruct the annotation path from the image relative path by
    replacing '.jpg' with '.lines.txt'.
    """
    anno_rel_path = img_rel_path.replace(".jpg", ".lines.txt")
    anno_dir = Path(data_root).joinpath(anno_rel_path)
    lanes: List[np.ndarray] = []
    with open(str(anno_dir), "r") as f:
        for line in f:
            coords_str = line.strip().split(" ")
            if len(coords_str) < 4:
                continue
            xs = []
            ys = []
            for i in range(0, len(coords_str), 2):
                xs.append(float(coords_str[i]))
                ys.append(float(coords_str[i + 1]))
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            lanes.append(pts)
    return lanes


def autoregressive_decode_tokens_v2(
    lanelm_model: LaneLMModel,
    visual_tokens: torch.Tensor,
    num_points: int,
    nbins_x: int,
    max_lanes: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same decoder as in test_lanelm_culane_v2, but for small batches."""
    device = visual_tokens.device
    B = visual_tokens.shape[0]
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

            if temperature and temperature > 0.0:
                probs_x = torch.softmax(step_logits_x / temperature, dim=-1)
                x_next = torch.multinomial(probs_x, num_samples=1).squeeze(1)
                probs_y = torch.softmax(step_logits_y / temperature, dim=-1)
                y_next = torch.multinomial(probs_y, num_samples=1).squeeze(1)
            else:
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

    x_tokens_all = torch.stack(all_x, dim=1)
    y_tokens_all = torch.stack(all_y, dim=1)
    return x_tokens_all, y_tokens_all


def coords_to_lane_v2(
    coords_resized: np.ndarray,
    ori_img_w: int,
    ori_img_h: int,
    crop_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Lane:
    if coords_resized.size == 0:
        return Lane(points=np.zeros((0, 2), dtype=np.float32))

    xs = coords_resized[:, 0]
    ys = coords_resized[:, 1]

    x_min, y_min, x_max, y_max = crop_bbox

    x_scale = float(ori_img_w) / float(img_w)
    y_scale = float(y_max - y_min) / float(img_h)

    x_orig = xs * x_scale
    y_orig = ys * y_scale + float(y_min)

    x_norm = x_orig / float(ori_img_w)
    y_norm = y_orig / float(ori_img_h)

    points = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    return Lane(points=points)


def draw_lanes_on_image(
    img_bgr: np.ndarray,
    gt_lanes: List[np.ndarray],
    pred_lanes: List[Lane],
    out_path: str,
) -> None:
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # Draw GT lanes in green
    for lane in gt_lanes:
        pts = lane.astype(np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    # Draw predicted lanes in red
    for lane in pred_lanes:
        if lane.points is None or lane.points.shape[0] == 0:
            continue
        pts = lane.points.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imwrite(out_path, img)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    test_cfg = cfg.model.get("test_cfg", {})
    ori_img_w = int(test_cfg.get("ori_img_w", 1640))
    ori_img_h = int(test_cfg.get("ori_img_h", 590))

    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        crop_bbox,
        img_scale,
    )

    img_w, img_h = img_scale

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset for vis (we will use its img_infos/annotations to load GT and images).
    dataset = build_culane_test_dataset_for_vis(
        data_root=args.data_root,
        list_path=args.test_list,
    )

    # Frozen CLRerNet
    clrernet_model = build_frozen_clrernet_backbone(
        config_path=args.config,
        checkpoint_path=args.clrernet_checkpoint,
        device=device,
    )

    # Load LaneLM v2
    ckpt = torch.load(args.lanelm_checkpoint, map_location="cpu")
    lm_cfg = ckpt.get("config", {})
    num_points = int(lm_cfg.get("num_points", 40))
    nbins_x = int(lm_cfg.get("nbins_x", 800))
    embed_dim = int(lm_cfg.get("embed_dim", 256))
    num_layers = int(lm_cfg.get("num_layers", 4))
    num_heads = int(lm_cfg.get("num_heads", 8))
    ffn_dim = int(lm_cfg.get("ffn_dim", 512))
    ckpt_max_lanes = int(lm_cfg.get("max_lanes", 4))

    max_lanes = min(args.max_lanes, ckpt_max_lanes)

    visual_in_channels = (64, 64, 64)
    max_y_tokens = num_points + 1
    max_seq_len = num_points * max_lanes * 2

    lanelm_model = LaneLMModel(
        nbins_x=nbins_x,
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

    tokenizer_cfg = LaneTokenizerConfig(
        img_w=img_w,
        img_h=img_h,
        num_steps=num_points,
        nbins_x=nbins_x,
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)

    # Visualize first N images
    n_vis = min(args.num_images, len(dataset))
    print(f"Visualizing {n_vis} images...")

    for idx in range(n_vis):
        data = dataset[idx]
        img_rel = dataset.img_infos[idx]  # relative path under data_root
        img_path = Path(args.data_root).joinpath(img_rel)

        # Load original BGR image in full resolution
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Load GT lanes in original coord system
        gt_lanes = load_gt_lanes_for_image(args.data_root, img_rel)

        # Prepare resized image tensor for LaneLM
        img_resized = data["img"]  # already cropped+resized by pipeline
        img_tensor = (
            torch.from_numpy(img_resized.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        # Visual tokens
        feats = extract_pyramid_feats(clrernet_model, img_tensor)
        visual_tokens = lanelm_model.encode_visual_tokens(feats)

        # Decode lanes
        x_tokens_all, y_tokens_all = autoregressive_decode_tokens_v2(
            lanelm_model=lanelm_model,
            visual_tokens=visual_tokens,
            num_points=num_points,
            nbins_x=nbins_x,
            max_lanes=max_lanes,
            temperature=args.temperature,
        )

        x_tokens_np = x_tokens_all.numpy()[0]  # (L, T)
        y_tokens_np = y_tokens_all.numpy()[0]  # (L, T)

        pred_lanes: List[Lane] = []
        for l in range(x_tokens_np.shape[0]):
            xt = x_tokens_np[l]
            yt = y_tokens_np[l]

            coords_resized = tokenizer.decode_single_lane(xt, yt)
            if coords_resized.shape[0] == 0:
                continue

            lane = coords_to_lane_v2(
                coords_resized=coords_resized,
                ori_img_w=ori_img_w,
                ori_img_h=ori_img_h,
                crop_bbox=tuple(crop_bbox),
                img_w=img_w,
                img_h=img_h,
            )
            if lane.points.shape[0] > 0:
                pred_lanes.append(lane)

        out_path = os.path.join(
            args.out_dir,
            f"vis_{idx:05d}.png",
        )
        draw_lanes_on_image(
            img_bgr=img_bgr,
            gt_lanes=gt_lanes,
            pred_lanes=pred_lanes,
            out_path=out_path,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
