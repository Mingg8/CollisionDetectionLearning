import numpy as np

config = {}
config["system_dim"] = 6
config["w_init"] = np.array([1, 3.0e-4, 1])
config["bound"] = {}
config["bound"]["min"] = np.array([-0.030, -0.030,  0.020, -1.00, -1.00, -11.00])
config["bound"]["max"] = np.array([0.030,  0.030,  0.070,  1.00,  1.00,  1.0])
# config["bound"]["min"] = np.array([-0.150, -0.150,  -0.150, -2.50, -2.50, -11.00])
# config["bound"]["max"] = np.array([0.150,  0.150,  0.150,  2.50,  2.50,  2.5])
config["m"] = 12
config["J"] = 0.8
# config["dt"] = 0.0005
# config["substep"] = 30 # limited to controller frequncy
config["dt"] = 0.001
config["substep"] =15 # limited to controller frequncy
config["high_level_step"] = 5
config["stabilization_coeff"] = 1.0e-10
config["damping_coeff"] = 5.0e-2
config["threshold"] = 0.20
# config["threshold"] = 0
config["no_contact_bound"] = {}
config["no_contact_bound"]["min"] = np.array([-0.030, -0.030,  0.047, -0.09, -0.09, -11.00])
config["no_contact_bound"]["max"] = np.array([0.030,  0.030,  0.070,  0.09,  0.09,  1.0])
config["goal"] = np.array([0.0, 0.0, 0.0386,  0, 0, -np.pi*3])
config["num_uniform_data"] = 200000
config["max_penetration_depth"] = 0.8/1000 # 1.5 mm
config["des_following_vel"] = 30 # travel nodes per s
config["tree_controller"] = {
    "receding_step" : 2,
    "purturbed_receding_step" : 4,
    # "stiffness" : np.diag([200,200,200,20,20,20]),
    # "stiffness" : np.diag([500,500,500,40,40,40]),
    "stiffness" : np.diag([1000,1000,1000,70,70,70]),
    # "stiffness" : np.diag([2000,2000,2000,100,100,100]),
    # "stiffness" : np.diag([4000,4000,4000,500,500,500]),
}
config["LQT"] = {
    "R":np.diag([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]), # running input cost
    "P":np.diag([5000, 5000, 5000, 5000, 5000, 5000, 1, 1, 1, 1, 1, 1]), # final state cost
    "Q":np.diag([5000, 5000, 5000, 5000, 5000, 5000, 1, 1, 1, 1, 1, 1]), # running state cost
    "nhorizon":10,
    "interval": 1,
    "vel_per_nodes" : 40
    # "P":np.diag([1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 1, 1, 1, 1]), # final state cost
    # "Q":np.diag([1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 1, 1, 1, 1]), # running state cost
    # "nhorizon":5,
    # "interval": 1,
    # "vel_per_nodes" : 40
}
