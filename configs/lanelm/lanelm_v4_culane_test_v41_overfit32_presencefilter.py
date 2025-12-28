_base_ = ['./lanelm_v4_culane_test.py']

# V41: evaluate presence-trained checkpoint with presence filtering enabled.
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v41_overfit32_presence/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        # Enable presence filtering now that presence head is trained.
        use_presence_filter=True,
        presence_threshold=0.5,
        smooth=False,
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,
        use_hr=True,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v41_test100_overfit32_presence/predictions',
)


