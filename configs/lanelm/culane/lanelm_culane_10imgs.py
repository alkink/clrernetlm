_base_ = ["../../clrernet/culane/dataset_culane_clrernet.py", "../../_base_/default_runtime.py"]

# MMDet scope
default_scope = 'mmdet'

# Import custom modules (must be after default_scope)
custom_imports = dict(
    imports=["libs.models", "libs.datasets", "libs.core.bbox", "libs.core.anchor", "libs.core.hook"],
    allow_failed_imports=False,
)

model = dict(
    type="LaneLMDetector",
    backbone=dict(
        type="DLANet",
        dla="dla34",
        pretrained=True,  # Use pretrained ImageNet weights
    ),
    neck=dict(
        type="CLRerNetFPN",
        in_channels=[128, 256, 512],  # DLA34 last 3 levels
        out_channels=64,
        num_outs=3,
    ),
    # Backbone/neck checkpoint will be loaded via load_from
    clrernet_checkpoint="clrernet_culane_dla34_ema.pth",
    # LaneLM-specific config (match train_lanelm_overfit_100.py)
    lanelm_cfg=dict(
        nbins_x=300,  # total vocab size (200 bins + 100 relative tokens)
        max_y_tokens=41,
        embed_dim=256,  # Match 100-img training
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=40,
        visual_in_channels=(64,),  # P5 only
        ckpt_path="work_dirs/lanelm_100_imgs/lanelm_100_best.pth",
    ),
    tokenizer_cfg=dict(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=200,  # tokenizer bins (absolute part)
        x_mode="relative_disjoint",
        max_abs_dx=32,  # relative part: [200, 265)
    ),
    decode_cfg=dict(
        max_lanes=4,
        temperature=0.0,
        crop_bbox=(0, 270, 1640, 590),
        ori_img_w=1640,
        ori_img_h=590,
        img_w=800,
        img_h=320,
    ),
    test_cfg=dict(),
    train_cfg=dict(),
)

# Disable training (inference-only)
train_dataloader = None
train_cfg = None
optim_wrapper = None
val_dataloader = None
val_cfg = None
val_evaluator = None

# Override test dataloader for 10 images
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_list="dataset/list/train_10.txt",  # Only 10 images
        test_mode=True,
    ),
)

# Use same evaluator as base config
test_evaluator = dict(
    type='CULaneMetric',
    data_root='dataset',
    data_list='dataset/list/train_10.txt',
)

# Test config
test_cfg = dict(type='TestLoop')

# Use MMDet standard visualizer
visualizer = None  # Disable visualizer completely for faster inference
