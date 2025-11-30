"""
Minimal test entrypoint for LaneLMDetector using mmengine Runner.

Key point:
- Do NOT use load_from; LaneLMDetector loads CLRerNet backbone/neck from its
  own config (clrernet_checkpoint) and LaneLM weights from lanelm_cfg.ckpt_path.
- This avoids mmengine trying to load CLRerNet head into LaneLMDetector.
"""

import argparse

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

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
