_base_ = ['./lanelm_v4_culane_test.py']

# V38 apples-to-apples: evaluate checkpoint trained with overfit-size=8 on 2k subset list.
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v38_train2k_overfit8_pad01/lanelm_v4_best.pth',
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
    result_dir='work_dirs/v38_test100_overfit8_pad01/predictions',
)


