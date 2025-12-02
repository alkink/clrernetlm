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
        nbins_x=200,  # CRITICAL: Must match training (train_lanelm_v4_fixed.py)
        max_y_tokens=41,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=80,
        # BU TEST KONFİGİ İÇİN: P5-Only LaneLM (train_lanelm_v4_fixed.py) kullanıyoruz
        # Eğitimde visual_in_channels=(64,) (sadece P5) idi, burada da aynısını kullanmalıyız.
        visual_in_channels=(64,),
        # TEST EDİLECEK MODEL: 1-image overfit + V5 mimarisi (work_dirs/lanelm_v4_fixed)
        ckpt_path='work_dirs/lanelm_v4_fixed/lanelm_v4_best.pth',
    ),
    tokenizer_cfg=dict(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=200,  # CRITICAL: Must match training (train_lanelm_v4_fixed.py line 204)
        x_mode='absolute',  # CRITICAL: Must match training (train_lanelm_v4_fixed.py line 224)
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

test_dataloader = dict(
    batch_size=128,  # Increased for faster inference
    num_workers=4,  # Increased for faster data loading (adjust if WSL issues)
    persistent_workers=True,  # Keep workers alive between epochs
    dataset=dict(
        data_root='dataset',
        data_list='dataset/list/train_100.txt',  # 100 training images - simpler, model has seen these
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
)

test_evaluator = dict(
    type='CULaneMetric',
    data_root='dataset',
    data_list='dataset/list/train_100.txt',  # 100 training images - simpler, model has seen these
    # Bu konfig: lanelm_v4_fixed checkpoint'i için sonuçları ayrı klasöre yazar
    result_dir='work_dirs/lanelm_v4_test_train100/predictions',
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
