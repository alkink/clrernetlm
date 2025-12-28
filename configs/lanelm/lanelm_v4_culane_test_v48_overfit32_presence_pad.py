_base_ = ['./lanelm_v4_culane_test.py']

# V48: A-çözümü (presence + pad/EOS öğretimi) ile eğitilmiş checkpoint'in testi.
# Amaç:
# - Lane sayısını presence filter ile düşürmek (FP azaltmak)
# - Lane uzunluğunu EOS/pad öğretimi ile stabilize etmek

model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v48_overfit32_presence_pad/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        # Lane count control
        use_presence_filter=True,
        presence_threshold=0.35,  # başlangıç; 0.3/0.4 sweep yapılabilir

        # Length control
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,

        # Keep eval strict
        smooth=False,
        use_hr=True,

        # Prompting kapalı: A çözümünün etkisini izole etmek için
        use_prompting=False,
    ),
)

# Not: test_lanelm_runner.py artık result_dir'i work_dir/predictions ile override ediyor.
# Bu satır sadece default fallback.
test_evaluator = dict(
    result_dir='work_dirs/v48_test100_overfit32_presence_pad/predictions',
)


