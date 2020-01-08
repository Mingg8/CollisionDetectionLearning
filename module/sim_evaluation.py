# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import os, sys
import DynamicsCpp
sys.path.insert(0, os.path.abspath("/home/dongwon/research/configuration_learning_nut"))
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjViewerBasic
import glfw
import numpy as np
from numpy import linalg as LA
import module.MLP as MLP
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from config import config
import random
# from plotly import graph_objs as go
# from multiprocessing import Pool
from copy import deepcopy
import module.util as util
from scipy.spatial.transform import Rotation as sR
from math import sin, cos, tan, asin, acos, atan, atan2
from module.mjc_module import MjcModule
# from module.dynamics import Dynamics
from module.tree import Tree

from datetime import datetime
from datetime import timedelta
import time

def initial_pos_distribution():
    state = np.random.normal(0, 0.004, [6]) * np.array([1,1,1,30,30,100]) + np.array([0,0,0.060,0,0,0])
    return state


if __name__=="__main__":       
        
    mjc = MjcModule('xmls/bolt_nut.xml',config)
    
    # construct MLP
    mlp = MLP.MLP(config)
    mlp.load_model('model_save/model7.h5')
    # mlp.export_weight()
    
    # initialize dynamics
    dyn_cpp = DynamicsCpp.Dynamics()
    dyn_cpp.change_param(config["dt"], config["threshold"], config["damping_coeff"], config["stabilization_coeff"])
    mlp.load_weights_to_dyn(dyn_cpp)
    # dyn_cpp.change_mass_recal(12,0.5)
    
    threshold = config["threshold"]
    damping_coeff = config["damping_coeff"]
    stabilization_coeff = config["stabilization_coeff"]
    
    tree = Tree("graph_data/graph_GraphML.xml", config)
       
    nsubstep = 10
    
    goal = config["goal"]
    
    # pos = sol_path[0]
    pos = initial_pos_distribution()
    vel = np.array([0,0,0,0,0,0])
    state_prior = np.zeros((12))
    state = np.array(list(pos) + list(vel))
    # remaining_path = tree.get_nearest_path(pos)
    step = 0
    substep = 0
    sim_time = 0
    traj_step = 0
    print_time = datetime.now()
    start_time = datetime.now()
    free_mode = False
    while True:
        ## reactive control
        if substep == 0:
            substep = config["substep"] ## test!
            policy_start_time = datetime.now()
            # u = tree.tree_follower(state)
            u = tree.LQT(state)
            policy_end_time = datetime.now()
        # u += 0.3*np.random.normal(0,1,[6]) * np.array([1,1,1,0.5,0.5,0.5])
        
        ## keyboard interface
        mag = 3/config["substep"]/config["dt"]
        # mjc.model.geom_rgba[0] = np.array([0.5,0.5,0.5,0.5])
        disturbance_color = np.array([0.761, 0.137, 0.149, 0.5])
        if mjc.viewer.key_press == glfw.KEY_A:
            u[1] -= mag*0.6
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_D:
            u[1] += mag*0.6
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_W:
            u[2] += mag*0.6
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_S:
            u[2] -= mag*0.6
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_E:
            u[5] -= mag*2
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_Q:
            u[5] += mag*2
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_Z:
            u[4] -= mag/2
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_C:
            u[4] += mag/2
            mjc.model.geom_rgba[0] = disturbance_color
        elif mjc.viewer.key_press == glfw.KEY_R:
            new_pos = initial_pos_distribution()
            new_vel = np.zeros([6])
            state = np.array(list(new_pos)+list(new_vel))
            traj_step = 0
        elif mjc.viewer.key_press == glfw.KEY_V: # for parameter tunning
            damping_coeff += 1.0e-3
            print(damping_coeff)
        elif mjc.viewer.key_press == glfw.KEY_B:
            damping_coeff -= 1.0e-3
            print(damping_coeff)
        elif mjc.viewer.key_press == glfw.KEY_F:
            stabilization_coeff += 1.0e-7
            print(stabilization_coeff)
        elif mjc.viewer.key_press == glfw.KEY_G:
            stabilization_coeff -= 1.0e-7
            print(stabilization_coeff)
        elif mjc.viewer.key_press == glfw.KEY_N: # for LQT tunning => LQT is more fit to this problem... use it!! and solve it
            damping_coeff += 1.0e-3
            print(damping_coeff)
        elif mjc.viewer.key_press == glfw.KEY_M:
            damping_coeff -= 1.0e-3
            print(damping_coeff)
        elif mjc.viewer.key_press == glfw.KEY_H:
            stabilization_coeff += 1.0e-7
        elif mjc.viewer.key_press == glfw.KEY_J:
            stabilization_coeff -= 1.0e-7
        elif mjc.viewer.key_press == glfw.KEY_Y:
            stabilization_coeff += 1.0e-6
        elif mjc.viewer.key_press == glfw.KEY_U:
            stabilization_coeff -= 1.0e-6
        elif mjc.viewer.key_press == glfw.KEY_X:
            free_mode = True
            print("free mode : ".format(free_mode))
            
        dyn_cpp.change_param(config["dt"], 0, damping_coeff, 1.0e-10)
        
        # fwd dynamics
        state_prior = state
        dyn_cpp.change_holonomic_contact_param(1, 0.00010, 0.3, 1) # dt 0.001 tunning
        
        # bolt_pos_uncertainty = np.array([0,0.002,-0.000,0,0,0])
        bolt_pos_uncertainty = np.array([0,0.000,0.000,0,0,0])
        state[:6] -= bolt_pos_uncertainty
        mjc.model.body_pos[2] = bolt_pos_uncertainty[:3]

        dyn_start_time = datetime.now()
        if free_mode :
            # state= dyn_cpp.fwd_dynamics_holonomic(state,np.array([0,0,-100,0,0,0]))
            state= dyn_cpp.fwd_dynamics_w_friction(state,np.array([0,0,-100,0,0,0]))
        else:
            # state = dyn_cpp.fwd_dynamics(state,u)
            # state = dyn_cpp.fwd_dynamics_holonomic(state,u)
            state = dyn_cpp.fwd_dynamics_w_friction(state,u)
        state[:6] += bolt_pos_uncertainty
        
        dyn_end_time = datetime.now()
        
        if dyn_end_time - print_time > timedelta(microseconds=100000):
            print_time = datetime.now()
            print("policy dt : {} // dyn dt : {} // sim t : {} // rt : {} // step : {} ".format(policy_end_time - policy_start_time,
                                                                                   dyn_end_time - dyn_start_time,
                                                                                   sim_time,
                                                                                   datetime.now() - start_time, traj_step))
            # print(dyn_cpp.get_fext())
        
        pos = state[:6]
        vel = state[6:]
        
        print(u)
        if np.linalg.norm(vel) > 10 :
            print(pos)
            print(vel)
            print("============")
        
        sim_time += config["dt"]
        mjc.viewer.key_press = "None"
        substep = substep - 1
        if substep == 0:
            traj_step += 1
            mjc.draw_arrow(4,5,pos[:3],pos[:3] + dyn_cpp.get_fext()[:3]/300)
            mjc.draw_arrow(6,7, pos[:3], pos[:3] + u[:3]/200)
            mjc.vis_cur_pos(pos)    
            mjc.model.geom_rgba[0] = np.array([0.5,0.5,0.5,0.5])
        
            while (datetime.now() - start_time) <= timedelta(microseconds=sim_time*1000000):
                time.sleep(0.00003)