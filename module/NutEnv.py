from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as LA
from copy import deepcopy
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.tune.logger import pretty_print
from ray.rllib.utils.memory import ray_get_and_free
from gym.spaces import Box

import module.util as util
from scipy.spatial.transform import Rotation as sR
from math import sin, cos, tan, asin, acos, atan, atan2

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
import time

import config as dyn_config
from module.MLP import MLP
import module.util as util
from module.dynamics import Dynamics
from module.mjc_module import MjcModule
from module.tree import Tree

dyn_config_ = dyn_config.config
tf = try_import_tf()   
    

class NutEnv_bc(gym.Env):
    def __init__(self, config):
        self.end_pos = config["end_pos"]
        init_pos = np.array([0, 0, 0.060, 0, 0, 0])
        init_vel = np.array([0, 0, 0, 0, 0, 0])
        self.state = np.array(list(init_pos) + list(init_vel))
        self.action_space = Box(low=np.array([-20.0, -20.0, -20.0, -10, -10, -10]),
                                high=np.array([20.0, 20.0, 20.0, 10, 10, 10]),
                                dtype=np.float32)
        # self.action_space = Box(low=dyn_config_["bound"]["min"],
        #                         high=dyn_config_["bound"]["max"],
        #                         dtype=np.float32)
        self.state_low = np.array(list(dyn_config_["bound"]["min"]) + [-1,-1,-1,-1,-1,-1] + [])
        self.state_high = np.array(list(dyn_config_["bound"]["max"]) + [1,1,1,1,1,1] + [])
        self.observation_space = Box(low=self.state_low,
                                high=self.state_high,
                                dtype=np.float32)
        # self.mjc = MjcModule('/home/dongwon/research/configuration_learning_nut/xmls/bolt_nut.xml',dyn_config_)
        self.traj_hist = []
        self.mjc = []
        self.mlp = MLP(dyn_config_)
        self.mlp.load_model('/home/dongwon/research/configuration_learning_nut/model_save')
        self.dyn = Dynamics(self.mlp)
        self.sub_itr = 5
        self.tree = Tree("/home/dongwon/research/configuration_learning_nut/graph_data/graph_GraphML.xml", dyn_config_)
        self.dyn_config = dyn_config_

    def state_clip(self):
        epsilone = 0.0001*np.ones([12])
        
        if (np.sum(self.state <= self.state_low) + np.sum(self.state >= self.state_high) != 0):
            self.state[np.where(self.state <= self.state_low)] = (self.state_low[np.where(self.state <= self.state_low)]
                                                                   + epsilone[np.where(self.state <= self.state_low)])
            self.state[np.where(self.state >= self.state_high)] = (self.state_high[np.where(self.state >= self.state_high)]
                                                                   - epsilone[np.where(self.state >= self.state_high)])
            return True
        return False
        
    def PID_controller(self, des_pos, des_vel):
        pos = self.state[:6]
        vel = self.state[6:]
        K = np.diag([4000,4000,4000,200,200,200])
        # K = np.diag([9000,9000,9000,600,600,600])
        # K = np.diag([10,10,10,1,1,1])
        M = self.dyn.M
        B = 2*np.sqrt(M*K)
        
        error_tran = pos[:3] - des_pos[:3]
        error_rot = util.XYZEuler2R(pos[3],pos[4],pos[5]) @ np.transpose(util.XYZEuler2R(des_pos[3],des_pos[4],des_pos[5]))
        error_rot_vec = sR.from_dcm(error_rot).as_rotvec()
        
        error = np.array(list(error_tran) + list(error_rot_vec))
        des_vel_rot_zero = np.zeros((6))
        des_vel_rot_zero[:3] = des_vel[:3]
        
        return -K@error - B@(vel - util.inv_Jacoxs(self.state)@des_vel)

    def reset(self):
        reset_pos = util.initial_pos_distribution()
        reset_vel = np.array([0,0,0,0,0,0])
        self.state = np.array(list(reset_pos) + list(reset_vel))
        self.state_clip()
        self.nstep = 0
        # self.traj_hist = []
        return self.state

    def step(self, action):
        
        action = np.array(action)      
        ref_u = self.tree.tree_follower(self.state)
        
        for _ in range(self.dyn_config["substep"]):
            input_u = deepcopy(action) - 8*self.state[6:]*np.array([1,1,1,0.05,0.05,0.05])
            # self.state = self.dyn.fwd_dynamics_wo_contact(self.state, input_u)
            self.state = self.dyn.compound_fwd_dynamics(self.state, input_u)
            self.state_clip()
            self.traj_hist.append(self.state)
            
        BC_reward = -LA.norm(ref_u - action)
        dist_from_goal = LA.norm(self.end_pos * np.array([1,1,1,0.01,0.01,0.01]) -self.state[:6] * np.array([1,1,1,0.01,0.01,0.01]))
        
        done = False
        self.nstep += 1
        if self.nstep >= 400 :
            done = True
        if (LA.norm(self.state[:6]-self.dyn_config['goal']) <  0.003):
            done = True
                
        input_energy = action @ self.dyn.M @ action
        c1 = 1/10
        c2 = 1/100
        # reward = c1*(-dist_from_goal) + c2*(-input_energy) + BC_reward/10
        reward = BC_reward/500
        return self.state, reward, done, {"BC_reward" : BC_reward, "dist_reward" : -c1*dist_from_goal, "input reward" : -c2*input_energy}

    def visualization_setting(self):
        self.mjc = MjcModule('/home/dongwon/research/configuration_learning_nut/xmls/bolt_nut.xml',dyn_config_)

    def visualize_traj(self):
        if self.mjc != [] :
            for j in range(len(self.traj_hist)-4000,len(self.traj_hist)):
                for i in range(6):
                    self.mjc.d.qpos[i] = self.traj_hist[j][i]
                self.mjc.sim.step()
                self.mjc.viewer.render()
                time.sleep(self.dyn.dt)
            
    def traj_reset(self):
        self.traj_hist = []




class NutEnv(gym.Env):
    def __init__(self, config):
        self.end_pos = config["end_pos"]
        init_pos = np.array([0, 0, 0.060, 0, 0, 0])
        init_vel = np.array([0, 0, 0, 0, 0, 0])
        self.state = np.array(list(init_pos) + list(init_vel))
        self.action_space = Box(low=np.array([-20.0, -20.0, -20.0, -10, -10, -10]),
                                high=np.array([20.0, 20.0, 20.0, 10, 10, 10]),
                                dtype=np.float32)
        # self.action_space = Box(low=dyn_config_["bound"]["min"],
        #                         high=dyn_config_["bound"]["max"],
        #                         dtype=np.float32)
        self.state_low = np.array(list(dyn_config_["bound"]["min"]) + [-20,-20,-20,-20,-20,-20])
        self.state_high = np.array(list(dyn_config_["bound"]["max"]) + [20,20,20,20,20,20])
        self.observation_space = Box(low=self.state_low,
                                high=self.state_high,
                                dtype=np.float32)
        # self.mjc = MjcModule('/home/dongwon/research/configuration_learning_nut/xmls/bolt_nut.xml',dyn_config_)
        self.traj_hist = []
        self.mjc = []
        self.mlp = MLP(dyn_config_)
        self.mlp.load_model('/home/dongwon/research/configuration_learning_nut/model_save')
        self.dyn = Dynamics(self.mlp)
        self.sub_itr = 25


    def state_clip(self):
        if (np.sum(self.state <= self.state_low) + np.sum(self.state >= self.state_high) != 0):
            self.state[np.where(self.state <= self.state_low)] = self.state_low[np.where(self.state <= self.state_low)]
            self.state[np.where(self.state >= self.state_high)] = self.state_high[np.where(self.state >= self.state_high)]
            return True
        return False
        
    def PID_controller(self, des_pos, des_vel):
        pos = self.state[:6]
        vel = self.state[6:]
        # K = np.diag([4000,4000,4000,200,200,200])
        # K = np.diag([9000,9000,9000,600,600,600])
        K = np.diag([10,10,10,1,1,1])
        M = self.dyn.M
        B = 2*np.sqrt(M*K)
        
        error_tran = pos[:3] - des_pos[:3]
        error_rot = util.XYZEuler2R(pos[3],pos[4],pos[5]) @ np.transpose(util.XYZEuler2R(des_pos[3],des_pos[4],des_pos[5]))
        error_rot_vec = sR.from_dcm(error_rot).as_rotvec()
        
        error = np.array(list(error_tran) + list(error_rot_vec))
        des_vel_rot_zero = np.zeros((6))
        des_vel_rot_zero[:3] = des_vel[:3]
        
        return -K@error - B@(vel - util.inv_Jacoxs(self.state)@des_vel)

    def reset(self):
        reset_pos = util.initial_pos_distribution()
        reset_vel = np.array([0,0,0,0,0,0])
        self.state = np.array(list(reset_pos) + list(reset_vel))
        self.state_clip()
        self.nstep = 0
        # self.traj_hist = []
        return self.state

    def step(self, action):
        
        action = np.array(action)
        # reward = 0
        # BC_reward = -LA.norm(PID_controller(self.end_pos, np.array([0,0,0,0,0,0]), self.state, self.dyn) - action)
        
        # u = self.PID_controller(action, np.array([0,0,0,0,0,0])) # for test
        
        for _ in range(self.sub_itr):
            input_u = deepcopy(action) - 5*self.state[6:]*np.array([1,1,1,0.05,0.05,0.05])
            self.state = self.dyn.fwd_dynamics_wo_contact(self.state, input_u)
            self.state_clip()
            self.traj_hist.append(self.state)
        dist_from_goal = LA.norm(self.end_pos * np.array([1,1,1,0.01,0.01,0.01]) -self.state[:6] * np.array([1,1,1,0.01,0.01,0.01]))
        done = False
        # if (dist_from_goal < 0.020):
            # done = True
            # reward = reward + 1
        
        self.nstep += 1
        if self.nstep >= 100 :
            done = True
            # reward = reward + 1
                
        input_energy = action @ self.dyn.M @ action
        # reward = (-dist_from_goal - 1/5*input_energy + BC_reward)/200
        reward = (-dist_from_goal) + (-input_energy)/10
        # if (np.sum(self.state <= self.state_low) + np.sum(self.state >= self.state_high) != 0):
        #     self.state[np.where(self.state <= self.state_low)] = self.state_low[np.where(self.state <= self.state_low)]
        #     self.state[np.where(self.state >= self.state_high)] = self.state_high[np.where(self.state >= self.state_high)]
        #     done = True
        # if self.state_clip():
            # done = True
        
        return self.state, reward, done, {"cur state" : list(self.state[:6])}

    def visualization_setting(self):
        self.mjc = MjcModule('/home/dongwon/research/configuration_learning_nut/xmls/bolt_nut.xml',dyn_config_)

    def visualize_traj(self):
        if self.mjc != [] :
            for j in range(len(self.traj_hist)-2000,len(self.traj_hist)):
                for i in range(6):
                    self.mjc.d.qpos[i] = self.traj_hist[j][i]
                self.mjc.sim.step()
                self.mjc.viewer.render()
                time.sleep(self.dyn.dt)
            
    def traj_reset(self):
        self.traj_hist = []