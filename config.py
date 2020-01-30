import numpy as np

config = {}
config["data_num"] = 1000000
config["epoch"] = 10
config["batch"] = 40
config["error_bound"] = 0.0005
config["weight"] = [1, 0.01]
config["hdim"] = 40 
config["layer_num"] = 3
config["trainset_ratio"] = 0.85
config["validset_ratio"] = 0.17

config["file_name"] = [
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat'
    ]

config["output_bound"] = [0, 1]
