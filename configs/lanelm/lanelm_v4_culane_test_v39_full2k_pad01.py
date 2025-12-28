_base_ = ['./lanelm_v4_culane_test.py']

# V39: evaluate checkpoint trained with full 2k subset (overfit-size=0) on test_100.
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v39_train2k_full_pad01/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        use_presence_filter=False,
        smooth=False,
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v39_test100_full2k_pad01/predictions',
)


