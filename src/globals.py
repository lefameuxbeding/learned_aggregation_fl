from haiku._src.data_structures import FlatMap

splitted_data = FlatMap({"image" : [[]], "label" : [[]]}) # Global to access from meta-training procedure, can probably find a better way to do this