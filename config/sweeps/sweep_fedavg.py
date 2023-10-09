_base_ = ["./sweeps_base.py"]

optimizer = "fedavg"
task = "image-mlp-fmst"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 4

sweep_config = dict(
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        local_learning_rate=dict(
            values=[
                1,
                0.5,
                0.3,
                0.1,
            ]
        ),
    ),
)
