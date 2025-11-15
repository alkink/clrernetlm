"""
Analyze LaneTokenizer quantization / de-quantization error on CULane.

This script:
  - Samples a subset of lanes from the CULane train set.
  - For each lane, fits the internal spline x(y), encodes it to tokens,
    decodes back to coordinates, and measures the x-coordinate error.
  - Reports aggregate statistics (mean / median / max error).

Run (inside conda 'clrernet' env):

PYTHONPATH=/home/alki/projects/clrernetlm:$PYTHONPATH \\
python tools/analyze_lanelm_tokenizer.py \\
  --data-root dataset \\
  --train-list dataset/list/train_gt.txt \\
  --num-images 200 \\
  --max-lanes-per-image 4
"""

import argparse
from typing import List

import numpy as np

from libs.datasets import CulaneDataset
from libs.datasets.pipelines import Compose
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig


def build_culane_raw_dataloader(
    data_root: str,
    list_path: str,
) -> CulaneDataset:
    """CULane dataset that only applies Albumentations pipeline, no packing.

    We mirror the training augmentation pipeline but stop before PackCLRNetInputs
    so that gt_points stays as raw polylines in the resized (800x320) space.
    """
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        train_al_pipeline,
    )

    train_pipeline = [
        dict(type="albumentation", pipelines=train_al_pipeline),
    ]

    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=train_pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=False,
    )
    return dataset


def analyze_tokenizer(
    data_root: str,
    train_list: str,
    num_images: int,
    max_lanes_per_image: int,
    img_w: int,
    img_h: int,
    num_steps: int,
    nbins_x: int,
) -> None:
    cfg = LaneTokenizerConfig(
        img_w=img_w,
        img_h=img_h,
        num_steps=num_steps,
        nbins_x=nbins_x,
    )
    tokenizer = LaneTokenizer(cfg)

    dataset = build_culane_raw_dataloader(data_root, train_list)
    n_samples = min(num_images, len(dataset))

    all_errors: List[float] = []

    print(f"Analyzing LaneTokenizer on {n_samples} images...")

    for idx in range(n_samples):
        data = dataset[idx]
        lanes = data.get("gt_points", [])
        # lanes: list of flat [x0,y0,x1,y1,...], already augmented and resized

        for li, lane_flat in enumerate(lanes[:max_lanes_per_image]):
            coords = np.array(lane_flat, dtype=np.float32).reshape(-1, 2)
            if coords.shape[0] < 2:
                continue

            # Fit spline and get continuous x(y) on tokenizer's sample_ys
            spline = tokenizer._fit_spline(coords)  # type: ignore[attr-defined]
            if spline is None:
                continue
            sample_ys = tokenizer._compute_sample_ys()  # type: ignore[attr-defined]
            xs_cont = spline(sample_ys)

            # Encode / decode using tokenizer
            x_tokens, y_tokens = tokenizer.encode_single_lane(coords)
            coords_dec = tokenizer.decode_single_lane(x_tokens, y_tokens)

            if coords_dec.shape[0] == 0:
                # Everything was padding / outside image
                continue

            # For decoded points, compute error vs continuous spline at same y's.
            # decode_single_lane reconstructs y positions from sample_ys.
            xs_dec = coords_dec[:, 0]
            ys_dec = coords_dec[:, 1]

            # Map decoded ys to nearest sample_ys index
            # (they should already match, but we are robust to float noise).
            idxs = np.clip(
                np.round(ys_dec / (img_h / num_steps)).astype(int),
                0,
                num_steps - 1,
            )
            xs_cont_aligned = xs_cont[idxs]

            err = np.abs(xs_dec - xs_cont_aligned)
            all_errors.extend(err.tolist())

    if not all_errors:
        print("No valid lanes found for analysis.")
        return

    errs = np.array(all_errors)
    print("LaneTokenizer x-coordinate error stats (in pixels):")
    print(f"  count   = {errs.size}")
    print(f"  mean    = {errs.mean():.4f}")
    print(f"  median  = {np.median(errs):.4f}")
    print(f"  max     = {errs.max():.4f}")
    print(f"  95th    = {np.percentile(errs, 95):.4f}")
    print(f"  99th    = {np.percentile(errs, 99):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LaneTokenizer quantization / de-quantization error on CULane."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="CULane dataset root.",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default="dataset/list/train_gt.txt",
        help="CULane train list file.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="Number of images to sample from train set.",
    )
    parser.add_argument(
        "--max-lanes-per-image",
        type=int,
        default=4,
        help="Max number of lanes per image to analyze.",
    )
    parser.add_argument(
        "--img-w",
        type=int,
        default=800,
        help="Tokenizer image width.",
    )
    parser.add_argument(
        "--img-h",
        type=int,
        default=320,
        help="Tokenizer image height.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40,
        help="Number of vertical samples T.",
    )
    parser.add_argument(
        "--nbins-x",
        type=int,
        default=800,
        help="Vocabulary size for x tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_tokenizer(
        data_root=args.data_root,
        train_list=args.train_list,
        num_images=args.num_images,
        max_lanes_per_image=args.max_lanes_per_image,
        img_w=args.img_w,
        img_h=args.img_h,
        num_steps=args.num_steps,
        nbins_x=args.nbins_x,
    )


if __name__ == "__main__":
    main()

