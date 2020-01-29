import numpy as np

config = {}
config["data_num"] = 1000
config["epoch"] = 10
config["batch"] = 40
config["error_bound"] = 0.0005
config["weight"] = [1, 0.01]
config["hdim"] = 64
config["split_ratio"] = [0.85, 0.15]

config["file_name"] = [
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat',
    '/obj/data/data_final_include_normal_modified.mat'
    ]