_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=1e-10,
    peak_value=1e-3,
    end_value=1e-4,
    warmup_steps=100,
    decay_steps=4900,
    exponent=1.0,
)
num_outer_steps = 5000
meta_loss_split = "outer_valid"
