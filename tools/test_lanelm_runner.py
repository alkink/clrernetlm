"""
Minimal test entrypoint for LaneLMDetector using mmengine Runner.

Key point:
- Do NOT use load_from; LaneLMDetector loads CLRerNet backbone/neck from its
  own config (clrernet_checkpoint) and LaneLM weights from lanelm_cfg.ckpt_path.
- This avoids mmengine trying to load CLRerNet head into LaneLMDetector.
"""

import argparse
import os
import shutil

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="LaneLM MMEngine test runner")
    parser.add_argument("config", help="config file")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="the directory to save evaluation metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for reproducible evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic algorithms where possible (can be slower).",
    )
    parser.add_argument(
        "--no-parallel-metric",
        action="store_true",
        help="Disable parallel (multiprocess) metric evaluation for determinism/debug.",
    )
    parser.add_argument(
        "--clean-preds",
        action="store_true",
        help="Delete prediction result_dir before running (prevents stale/mixed files).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # Ensure load_from is disabled; weights are loaded internally by LaneLMDetector
    cfg.load_from = None
    # Avoid multiprocessing issues in restricted environments
    if cfg.get("test_dataloader", None):
        cfg.test_dataloader["num_workers"] = 0
        cfg.test_dataloader["persistent_workers"] = False
    if cfg.get("env_cfg", None) and "mp_cfg" in cfg.env_cfg:
        cfg.env_cfg["mp_cfg"]["mp_start_method"] = "fork"
    if args.work_dir:
        cfg.work_dir = args.work_dir

    # --- Make result_dir unique per run to avoid overwrites across experiments ---
    # Many configs previously pointed to shared paths like work_dirs/v42_.../predictions.
    # That makes it very easy to accidentally compare metrics from different runs.
    work_dir = cfg.get("work_dir", None) or args.work_dir
    if work_dir:
        pred_dir = os.path.join(str(work_dir), "predictions")
        if cfg.get("test_evaluator", None) and isinstance(cfg.test_evaluator, dict):
            cfg.test_evaluator["result_dir"] = pred_dir
            if args.no_parallel_metric:
                cfg.test_evaluator["use_parallel"] = False
        if args.clean_preds:
            shutil.rmtree(pred_dir, ignore_errors=True)
        os.makedirs(pred_dir, exist_ok=True)

    # --- Reproducibility knobs ---
    # MMEngine uses cfg.randomness if present.
    cfg.randomness = dict(seed=int(args.seed), deterministic=bool(args.deterministic))

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
