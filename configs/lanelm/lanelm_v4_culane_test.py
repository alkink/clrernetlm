_base_ = [
    '../clrernet/culane/dataset_culane_clrernet.py',
    '../_base_/default_runtime.py',
]

# IMPORTANT: do not load any external checkpoint via load_from; LaneLMDetector
# loads CLRerNet backbone and LaneLM ckpt internally.
load_from = None
custom_hooks = []
default_scope = 'mmdet'

custom_imports = dict(
    imports=["libs.models", "libs.datasets", "libs.core.bbox", "libs.core.anchor", "libs.core.hook"],
    allow_failed_imports=False,
)

model = dict(
    type='LaneLMDetector',
    backbone=dict(type='DLANet', dla='dla34', pretrained=True),
    neck=dict(type='CLRerNetFPN', in_channels=[128, 256, 512], out_channels=64, num_outs=3),
    clrernet_checkpoint='clrernet_culane_dla34_ema.pth',
    lanelm_cfg=dict(
        nbins_x=800,  # V7: PDF standard (line 570: "800 nbins and 100 training epochs")
        # NOTE: Model must be retrained with nbins_x=800 to match this config
        max_y_tokens=41,
        embed_dim=512,  # V17: PDF'de LaneLM-512 (DLA34) için 512 (önceden 256 yanlıştı!)
        num_layers=3,  # V15: PDF'ye göre 3 layers (line 382: "consists of 3 layers of LaneLM blocks")
        num_heads=8,
        ffn_dim=512,
        max_seq_len=80,
        # V18: PDF'de Full FPN (P3+P4+P5) kullanılıyor (Line 344-365, Table 5)
        # PDF Ablation Study: FPN yok: 68.36, FPN var: 70.71 (+2.35 F1!)
        # Eğitimde visual_in_channels=(64, 64, 64) (Full FPN) kullanılıyor, burada da aynısını kullanmalıyız.
        visual_in_channels=(64, 64, 64),  # V18: Full FPN (P3+P4+P5)
        # TEST EDİLECEK MODEL: 1-image overfit + V5 mimarisi (work_dirs/lanelm_v4_fixed)
        ckpt_path='work_dirs/v22_overfit1_0kp/lanelm_v4_best.pth',
    ),
    tokenizer_cfg=dict(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=800,  # V7: PDF standard (must match training)
        x_mode='absolute',  # CRITICAL: Must match training (train_lanelm_v4_fixed.py line 236)
    ),
    decode_cfg=dict(
        max_lanes=4,
        temperature=0.0,
        crop_bbox=(0, 270, 1640, 590),
        ori_img_w=1640,
        ori_img_h=590,
        img_w=800,
        img_h=320,
        # V25: Overfit / debug checkpoint'lerinde presence head genelde eğitilmedi (presence_weight=0).
        # Bu durumda presence_filter rastgele lane eler ve görselleri bozar. Debug için kapalı tut.
        use_presence_filter=False,
        presence_threshold=0.3,
        use_prompting=False,  # V20A: 0-kp test, prompting kapalı
    ),
    test_cfg=dict(),
    train_cfg=dict(),
)

test_dataloader = dict(
    batch_size=128,  # Increased for faster inference
    num_workers=4,  # Increased for faster data loading (adjust if WSL issues)
    persistent_workers=True,  # Keep workers alive between epochs
    dataset=dict(
        data_root='dataset',
        data_list='dataset/list/test_100.txt',  # 100 images - quick test subset
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
)

test_evaluator = dict(
    type='CULaneMetric',
    data_root='dataset',
    data_list='dataset/list/test_100.txt',  # 100 images - quick test subset
    # Bu konfig: lanelm_v4_fixed checkpoint'i için sonuçları ayrı klasöre yazar
    result_dir='work_dirs/lanelm_v4_test_fixed_100/predictions',
    use_parallel=True,  # Enable parallel for faster evaluation on large dataset
)

test_cfg = dict(type='TestLoop')
train_cfg = None
train_dataloader = None
val_cfg = None
val_dataloader = None
val_evaluator = None
optim_wrapper = None
visualizer = None
