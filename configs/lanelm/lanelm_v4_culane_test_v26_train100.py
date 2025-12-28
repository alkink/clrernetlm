_base_ = [
    './lanelm_v4_culane_test.py',
]

# V26: train_100 üzerinde gerçek eğitim checkpoint'i ile test_100 çalıştırma config'i.
# Not: presence head bu koşuda opsiyonel; presence loss verilmediyse presence filter kapalı tutulmalı.

model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v26_train100_0kp/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        use_presence_filter=False,
        presence_threshold=0.3,
        # V28: make tokenizer smoothing explicit; default False to avoid degrading strict IoU.
        smooth=False,
        # V36: enable EOS stop (x=0) to avoid drawing/metric pollution from padding regions.
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v26_test100/predictions',
)



