# must be overriden
run_type = None

# common
optimizer = "fedavg"
task = "small-image-mlp-fmst"
hidden_size = 32
local_learning_rate = 0.5
local_batch_size = 80
num_grads = 10
num_local_steps = 4
num_inner_steps = 1000
learning_rate = 0.0001
name_suffix = ""

# meta training only
num_outer_steps = 5000
from_checkpoint = False
num_devices = 1
use_pmap = False
auto_resume = False
meta_loss_split = None

# meta testing only
num_runs = 10
wandb_checkpoint_id = None
test_project = "learned_aggregation_fl"

# for slowmo
beta = 0.99

# FL
number_clients = 100
participation_rate = 0.1

# sweeps only
sweep_config = dict()
