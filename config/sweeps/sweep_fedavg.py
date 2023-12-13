_base_ = ["./sweeps_base.py"]

sweep_config = dict(
    method="grid",
    metric=dict(name="valid loss", goal="minimize"),
    parameters=dict(
        alpha=dict(
            values=[
                0.1,
                100,
            ]
        ),
        local_learning_rate=dict(
            values=[
                1,
                0.1,
                0.01,
                0.001,
                0.0001,
            ]
        ),
        global_learning_rate=dict(
            values=[
                1,
                0.1,
                0.01,
                0.001,
                0.0001,
            ]
        ),
    ),
)
