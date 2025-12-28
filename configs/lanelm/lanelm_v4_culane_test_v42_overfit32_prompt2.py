_base_ = ['./lanelm_v4_culane_test.py']

# V42: paper-aligned CULane "*" prompting (2 adjacent keypoints from CLRNet).
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v42_overfit32_prompt2/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        # Enable CLRNet prompting at inference (paper "two initial keypoints")
        use_prompting=True,
        # Presence filter OFF initially to avoid confounding until we confirm prompting works.
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
    result_dir='work_dirs/v42_test100_overfit32_prompt2/predictions',
)


