from haiku._src.data_structures import FlatMap

splitted_data = FlatMap({"image" : [[]], "label" : [[]]}) # Global to access from meta-training procedure, can probably find a better way to do this

num_local_steps = 4
number_clients = 100
participation_rate = 0.1
local_batch_size = 80