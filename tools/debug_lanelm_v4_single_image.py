"""
LaneLM v4 Single-Image Debug Script
===================================

Amaç:
- `train_lanelm_v4_fixed.py` ile overfit edilen *tek bir görüntü* üzerinde
  **GT token'ları** ile **model tahmin token'larını** birebir karşılaştırmak.
- Özellikle belirli bir lane (örn. sağ sarı) için:
  - Her adımda (t) GT X token, Pred X token
  - Decode sonrası piksel uzayında X farkı (|pred_x - gt_x|)
  - Alt / orta / üst bölgeler için ayrı hata istatistikleri

Bu script, tahminlerin neden zigzag olduğunu *tahmine dayanmadan* sayısal olarak
görmemizi sağlar.
"""

import argparse
import os

import cv2
import numpy as np
import torch

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig, LaneLMModel
from libs.models.detectors.lanelm_detector import autoregressive_decode as visual_first_decode
from tools.train_lanelm_culane_v3 import (
    LaneLMHyperParams,
    collate_lanelm_batch_v3,
    build_frozen_clrernet_backbone,
)
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg,
    crop_bbox,
    img_scale,
)


def extract_p5_feat(model, imgs: torch.Tensor):
    """P5-only feature extraction (aynı train_v4_fixed'teki gibi)."""
    with torch.no_grad():
        feats = model.extract_feat(imgs)
        p5 = feats[-1]
    return [p5]


def build_lanelm_model_v4_debug(hparams, visual_in_channels):
    """Train script ile aynı LaneLM v4 modelini kur."""
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
    return model


def build_clean_dataloader_single(data_root, list_path):
    """Train script ile aynı clean pipeline, ama tek görüntü (index=0)."""
    clean_pipeline = [
        dict(type="Compose", params=compose_cfg),
        dict(
            type="Crop",
            x_min=crop_bbox[0],
            x_max=crop_bbox[2],
            y_min=crop_bbox[1],
            y_max=crop_bbox[3],
            p=1,
        ),
        dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    ]
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,
    )
    # Sadece ilk görüntü
    from torch.utils.data import Subset

    subset = Subset(dataset, [0])
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_lanelm_batch_v3,
    )
    return dataloader, dataset


def debug_single_image(
    config_path: str,
    backbone_ckpt: str,
    lanelm_ckpt: str,
    data_root: str,
    list_path: str,
    device: torch.device,
    lane_idx: int,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    # 1) Dataset & dataloader
    dataloader, dataset = build_clean_dataloader_single(data_root, list_path)
    batch = next(iter(dataloader))

    # 2) CLRerNet backbone (frozen)
    print("Loading CLRerNet backbone (frozen) for debug...")
    clrernet = build_frozen_clrernet_backbone(config_path, backbone_ckpt, device)

    # 3) LaneLM v4 model (P5 Only + Absolute)
    nbins_x = 200
    hparams = LaneLMHyperParams(
        nbins_x=nbins_x,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4,
    )
    visual_in_channels = (64,)  # P5 Only

    lanelm = build_lanelm_model_v4_debug(hparams, visual_in_channels).to(device)

    print(f"Loading LaneLM v4 weights from {lanelm_ckpt} ...")
    ckpt = torch.load(lanelm_ckpt, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = lanelm.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    lanelm.eval()

    # 4) Tokenizer (ABSOLUTE, nbins_x=200)
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=nbins_x,
        x_mode="absolute",
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)

    # 5) GT tokenlarını oluştur (aynı train script mantığı)
    imgs = batch["inputs"].to(device)
    with torch.no_grad():
        feats = extract_p5_feat(clrernet, imgs)
        visual_tokens_full = lanelm.encode_visual_tokens(feats)  # (B, N, D)

    # Tek görüntü, tüm lane'ler
    gt_points_all = batch["gt_points"][0]
    gt_points_all = [l for l in gt_points_all if len(l) >= 4][: hparams.max_lanes]
    if lane_idx >= len(gt_points_all):
        print(
            f"[WARN] lane_idx={lane_idx} için GT lane yok. "
            f"Toplam GT lane sayısı: {len(gt_points_all)}"
        )
        return

    # Seçilen lane için GT tokenları
    pts = np.array(gt_points_all[lane_idx], dtype=np.float32).reshape(-1, 2)
    gt_x_tokens_np, gt_y_tokens_np = tokenizer.encode_single_lane(pts)

    # Teacher forcing input (training ile aynı)
    x_tokens = torch.from_numpy(gt_x_tokens_np).unsqueeze(0).to(device)  # (1, T)
    y_tokens = torch.from_numpy(gt_y_tokens_np).unsqueeze(0).to(device)  # (1, T)

    x_in = x_tokens.clone()
    x_in[:, 1:] = x_tokens[:, :-1]
    x_in[:, 0] = x_tokens[:, 0]  # training ile hizalı başlangıç

    y_in = y_tokens.clone()
    y_in[:, 1:] = y_tokens[:, :-1]
    y_in[:, 0] = y_tokens[:, 0]

    lane_ids = torch.tensor([lane_idx], dtype=torch.long, device=device)

    # 6) Teacher forcing ile logits al (tam sequence)
    with torch.no_grad():
        logits_x, _ = lanelm(visual_tokens_full, x_in, y_in, lane_indices=lane_ids)
        # shape: (1, T, nbins_x)
        pred_x_tokens_tf = torch.argmax(logits_x, dim=-1)[0].cpu().numpy()  # (T,)

    # 7) Autoregressive (visual-first) decoding - training/test ile aynı mantık
    with torch.no_grad():
        x_tokens_all, y_tokens_all = visual_first_decode(
            lanelm_model=lanelm,
            visual_tokens=visual_tokens_full,
            tokenizer_cfg=tokenizer_cfg,
            max_lanes=hparams.max_lanes,
        )
    # Tek görüntü, seçilen lane
    pred_x_tokens_ar = x_tokens_all[0, lane_idx].cpu().numpy()
    pred_y_tokens_ar = y_tokens_all[0, lane_idx].cpu().numpy()

    # 8) Decode to pixel coords
    gt_coords = tokenizer.decode_single_lane(gt_x_tokens_np, gt_y_tokens_np, smooth=False)

    # Teacher forcing decode (model'in token çıkışı, ama GT Y ile)
    tf_coords = tokenizer.decode_single_lane(pred_x_tokens_tf, gt_y_tokens_np, smooth=False)

    # Autoregressive decode (visual-first yolu)
    ar_coords = tokenizer.decode_single_lane(
        pred_x_tokens_ar, pred_y_tokens_ar, smooth=False
    )

    # 9) Hata analizi (per-step X farkı)
    def compute_x_error(ref, pred, name: str):
        if len(ref) == 0 or len(pred) == 0:
            print(f"[{name}] Boş koordinat; analiz yapılamıyor.")
            return
        # Y'ye göre hizala: her sample_ys t için bir nokta var, sıralı
        min_len = min(len(ref), len(pred))
        ref = ref[:min_len]
        pred = pred[:min_len]
        diff = np.abs(ref[:, 0] - pred[:, 0])
        mean_err = float(diff.mean())
        bottom_th = int(0.33 * min_len)
        mid_th = int(0.66 * min_len)
        bottom_err = float(diff[:bottom_th].mean()) if bottom_th > 0 else float("nan")
        mid_err = (
            float(diff[bottom_th:mid_th].mean())
            if mid_th > bottom_th
            else float("nan")
        )
        top_err = (
            float(diff[mid_th:].mean()) if min_len > mid_th else float("nan")
        )
        print(
            f"[{name}] mean_err={mean_err:.2f}px | "
            f"bottom={bottom_err:.2f}px mid={mid_err:.2f}px top={top_err:.2f}px"
        )
        # Ayrıntılı tablo
        print(f"\n{name} per-step X error (first 20 steps):")
        for i in range(min(20, min_len)):
            print(
                f"  t={i:02d}  y={ref[i,1]:6.1f}  "
                f"GT_X={ref[i,0]:7.1f}  PR_X={pred[i,0]:7.1f}  "
                f"|Δx|={diff[i]:5.1f}"
            )

    print(f"\n=== Lane {lane_idx} Debug (Single Image) ===")
    compute_x_error(gt_coords, tf_coords, "TeacherForcing")
    compute_x_error(gt_coords, ar_coords, "Autoregressive")

    # 10) Görsel çıktı (GT / TF / AR birlikte)
    img_vis = (imgs[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    def draw_lane(img, coords, color, thickness=2):
        if coords is None or len(coords) < 2:
            return
        pts = coords.astype(int)
        H, W = img.shape[:2]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            if 0 <= x1 < W and 0 <= x2 < W and 0 <= y1 < H and 0 <= y2 < H:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # GT: yeşil, Teacher forcing: mavi, Autoregressive: kırmızı
    draw_lane(img_vis, gt_coords, (0, 255, 0), 3)
    draw_lane(img_vis, tf_coords, (255, 0, 0), 2)
    draw_lane(img_vis, ar_coords, (0, 0, 255), 2)

    out_path = os.path.join(save_dir, f"debug_lane{lane_idx}.jpg")
    cv2.imwrite(out_path, img_vis)
    print(f"\nSaved debug visualization to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/clrernet/culane/clrernet_culane_dla34_ema.py",
    )
    parser.add_argument(
        "--backbone-ckpt",
        default="clrernet_culane_dla34_ema.pth",
        help="CLRerNet backbone checkpoint",
    )
    parser.add_argument(
        "--lanelm-ckpt",
        default="work_dirs/lanelm_v4_fixed/lanelm_v4_best.pth",
        help="LaneLM v4 checkpoint (train_lanelm_v4_fixed.py çıktısı)",
    )
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument(
        "--list-path",
        default="dataset/list/train_100.txt",
        help="train_lanelm_v4_fixed.py ile kullanılan liste",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lane-idx",
        type=int,
        default=3,
        help="Debug edilecek lane index'i (0-3, genelde sağ sarı için 3)",
    )
    parser.add_argument(
        "--save-dir",
        default="work_dirs/lanelm_v4_fixed/debug_single",
        help="Çıktı görselleri ve loglar",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    debug_single_image(
        config_path=args.config,
        backbone_ckpt=args.backbone_ckpt,
        lanelm_ckpt=args.lanelm_ckpt,
        data_root=args.data_root,
        list_path=args.list_path,
        device=device,
        lane_idx=args.lane_idx,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()


