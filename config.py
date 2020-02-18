import numpy as np

config = {}

config["input_num"] = 6
config["output_num"] = 1
config["input_file_name"] = 'q_set'
config["output_file_name"] = 'penet_min'
config["data_num"] = 300000


config["error_bound"] = 0.0005
config["weight"] = [1, 0.01]
config["trainset_ratio"] = 0.85
config["validset_ratio"] = 0.17

config["hdim"] = 64
config["layer_num"] = 2

config["epoch"] = 20
config["batch"] = 40


config["file_name"] = [
    '/obj/data/total_data_only_penet.mat'
    ]

config["output_bound"] = [0, 1]