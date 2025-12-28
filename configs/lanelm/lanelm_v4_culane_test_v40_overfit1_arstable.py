_base_ = ['./lanelm_v4_culane_test.py']

# V40 apples-to-apples: evaluate checkpoint trained with overfit-size=1 (AR-stable settings).
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v40_overfit1_arstable/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        use_presence_filter=False,
        smooth=False,
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,
        use_hr=True,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v40_test100_overfit1_arstable/predictions',
)


