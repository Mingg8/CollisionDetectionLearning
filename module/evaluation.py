# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import os, sys
sys.path.insert(0, os.path.abspath("/home/dongwon/research/configuration_learning_nut"))
import numpy as np
from numpy import linalg as LA
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from config import config
import random
from plotly import graph_objs as go
from multiprocessing import Pool
from copy import deepcopy
from module.util import collision_free_config_gen, random_state_generation, random_states_generation, R2XYZEuler, XYZEuler2R, R_diff
from config import config
from module.plotting import Plot
from scipy.spatial.transform import Rotation as sR
from math import sin, cos, tan, asin, acos, atan, atan2


import multiprocessing
multiprocessing.set_start_method('spawn', True)

# should implement 2 plot

## first plot for evaluation
# plot near CF path with fixed SO3
def plot_near_CF_path_pred(mlp):
    print("plot near collision free path")
    
    resolution = 10000
    total_angle = 4*np.pi
    states = []
    for i in range(resolution):
        states.append(collision_free_config_gen(-i/resolution*total_angle, output_type=1))
    
    states += (list(np.array(states) + np.array([0,0,0.00017,0,0,0]))+
        list(np.array(states) + np.array([0,0,-0.00011,0,0,0]))+
        list(np.array(states) + np.array([0.0002,0,0,0,0,0]))+
        list(np.array(states) + np.array([-0.0002,0,0,0,0,0]))
        )

    pred = mlp.predict(np.array(states))
    
    states = np.array(states)
    r = 0.001
    for i in range(len(list(states))):
        states[i,:3] += np.array([r*cos(-i/resolution*total_angle), r*sin(-i/resolution*total_angle), 0])
    
    plot = Plot("test")
    plot.plot_3d_pnts(states, colors = pred)
    plot.draw()
    
## second plot for evaluation
# plot all space with fixed SO3
def plot_R3_space_pred(mlp,R):
    print("plot_R3_space_pred")
    bound = config["bound"]
    sample_num = 50000
    sample_states = random_states_generation(sample_num)
    sample_states[:,3:] = np.ones(np.shape(sample_states[:,3:])) * R2XYZEuler(R) + np.array([0,0,-0.1])
    print("sample num : {}, rot state : {}".format(sample_num, R2XYZEuler(R).transpose()))
    
    sample_states = list(sample_states)
    
    # argumenting yellow data
    sample_states += [collision_free_config_gen(-0.1, output_type=1)]
    sample_states += [collision_free_config_gen(-0.1-2*np.pi, output_type=1)]
    
    sample_states = np.array(sample_states)
    
    pred = mlp.predict(sample_states)
    
    plot = Plot("test")
    plot.plot_3d_pnts(sample_states, colors = pred)
    # plot.plot_2d_pnts(pred)
    plot.draw()
    
def plot_R3_space_input(LIO, R):
    print("plot R3 space")
    bound = config["bound"]
    
    input = LIO["input"]
    output = LIO["output"]
    print("sample num : {}".format(len(input)))
    
    plot_input = []
    plot_output = []
    euler = R2XYZEuler(R)
    for i in range(len(input)):
        if R_diff(R,XYZEuler2R(input[i][3], input[i][4],input[i][5])) < 0.01 :
        # if LA.norm((np.array(input[i][3:]) - euler)) < 0.1:
            plot_input.append(input[i])
            plot_output.append(output[i])
    
    print("plot sample num : {}".format(len(plot_input)))
    plot_input = np.array(plot_input)
    plot_output = np.array(plot_output)
    
    plot_max_num = 10000
    plot = Plot("test")
    plot.plot_3d_pnts(plot_input[:plot_max_num,:3], colors = plot_output[:plot_max_num])
    # plot.plot_2d_pnts(pred)
    plot.draw()
    
def plot_near_CF_path_fcl(cd):
    resolution = 1000
    total_angle = 4*np.pi
    cd_input = {}
    cd_input['tran'] = []
    cd_input['rot'] = []
    states = []
    for i in range(resolution):
        tran_tmp, rot_tmp = collision_free_config_gen(-i/resolution*2*total_angle)
        states.append(collision_free_config_gen(-i/resolution*2*total_angle, output_type = 1))
        cd_input['tran'].append(tran_tmp)
        cd_input['rot'].append(rot_tmp)
    
    cd_input['tran'] += (list(np.array(cd_input['tran']) + np.array([0,0,0.00017]))+
        list(np.array(cd_input['tran']) + np.array([0,0,-0.00011]))+
        list(np.array(cd_input['tran']) + np.array([0.0002,0,0]))+
        list(np.array(cd_input['tran']) + np.array([-0.0002,0,0]))
        )
    cd_input['rot'] += (list(np.array(cd_input['rot']))+
        list(np.array(cd_input['rot']))+
        list(np.array(cd_input['rot']))+
        list(np.array(cd_input['rot']))
        )
    
    states += (list(np.array(states) + np.array([0,0,0.0001,0,0,0]))+
        list(np.array(states) + np.array([0,0,-0.0001,0,0,0]))+
        list(np.array(states) + np.array([0.0001,0,0,0,0,0]))+
        list(np.array(states) + np.array([-0.0001,0,0,0,0,0]))
        )
    
    cd_result = []
    for i in range(len(cd_input['tran'])):
        cd_result.append(cd.collisioncheck(cd_input['rot'][i], cd_input['tran'][i]))
    
    r = 0.001
    states = np.array(states)
    for i in range(len(list(states))):
        states[i,:3] += np.array([r*cos(-i/resolution*2*total_angle), r*sin(-i/resolution*2*total_angle), 0])
    
    plot = Plot("test")
    plot.plot_3d_pnts(np.array(states), colors = np.array(cd_result))
    # plot.plot_2d_pnts(pred)
    plot.draw()
    
    

if __name__=="__main__":       
    # construct MLP
    mlp = MLP.MLP(config)
    
    # load model
    mlp.load_model('model_save/model7.h5')
    
    # # test
    # pos = np.array([0,0,0.050, 0, 0, 0])
    # pos_ex = np.expand_dims(pos, axis=1)
    # mlp.predict(pos_ex)
    
    # evaluation 1st
    draw_R = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    plot_R3_space_pred(mlp, draw_R)
    plot_near_CF_path_pred(mlp)
    
    
    # with open('./data_save/LIO_data.pkl', 'rb') as f:
    #     LIO = pickle.load(f)
 
    # iR = sR.from_rotvec([0,0,0]).as_dcm()
    # print(iR)
    # plot_R3_space_input(LIO,iR)
 
    
    
    
    
    
    
    