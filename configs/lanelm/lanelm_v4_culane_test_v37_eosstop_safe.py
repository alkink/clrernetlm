_base_ = [
    './lanelm_v4_culane_test.py',
]

# V37: EOS-stop enabled, with minimum-step guards to avoid stopping at t=0.
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v36_train100_tokenfix_padloss/lanelm_v4_best.pth',
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

# IMPORTANT: write predictions to a dedicated folder (avoid overwriting older runs).
test_evaluator = dict(
    result_dir='work_dirs/v37_test100_eosstop_safe/predictions',
)


