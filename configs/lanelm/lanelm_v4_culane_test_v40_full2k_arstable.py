_base_ = ['./lanelm_v4_culane_test.py']

# V40: evaluate AR-stabilized full-2k checkpoint on test_100.
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v40_train2k_full_arstable/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        use_presence_filter=False,
        smooth=False,
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,
        # keep HR ON (paper default); we already verified HR isn't the root cause in V39.
        use_hr=True,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v40_test100_full2k_arstable/predictions',
)


