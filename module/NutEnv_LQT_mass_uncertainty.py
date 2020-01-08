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
# from module.MLP import MLP
import module.util as util
# from module.dynamics import Dynamics
import DynamicsCpp
from module.mjc_module import MjcModule
from module.tree import Tree

dyn_config_ = dyn_config.config
tf = try_import_tf()   
    

class NutEnv_LQT(gym.Env):
    def __init__(self, config):
        self.end_pos = config["end_pos"]
        init_pos = np.array([0, 0, 0.060, 0, 0, 0])
        init_vel = np.array([0, 0, 0, 0, 0, 0])
        self.state = np.array(list(init_pos) + list(init_vel))
        self.fext = np.zeros((6))
        self.ref_poss = np.zeros((18))
        self.goal_wrt_pos = self.end_pos - self.state[:6]

        ## definition of action space : Q_tran, Q_e, Q_v, Q_de, interval, node_vel, horizon, FF_input(6)
        Q_tran_low = 100
        Q_tran_high = 11000
        Q_e_low = 100
        Q_e_high = 11000
        Q_v_low = 1
        Q_v_high = 1000
        Q_de_low = 1
        Q_de_high = 1000
        interval_low = 1
        interval_high = 5
        node_vel_low = 5
        node_vel_high = 120
        horizon_low = 2
        horizon_high = 13
        FF_input_low = np.array([-20.0, -20.0, -20.0, -8, -8, -8])
        FF_input_high = np.array([20.0, 20.0, 20.0, 8, 8, 8])
        self.action_low = np.array([Q_tran_low, Q_e_low, Q_v_low, Q_de_low, interval_low, node_vel_low, horizon_low] + list(FF_input_low))
        self.action_high = np.array([Q_tran_high, Q_e_high, Q_v_high, Q_de_high, interval_high, node_vel_high, horizon_high] + list(FF_input_high))
        self.action_space = Box(low=-1*np.ones([13]),
                            high=1*np.ones([13]),
                            dtype=np.float32)

        ## definition of observation space : cur_state(12) + fext(6) + ref_traj_wrt_cur_pos(6*3) + goal_wrt_cur_pos(6)
        self.state_low = np.array(list(dyn_config_["bound"]["min"]) + [-1,-1,-1,-1,-1,-1.5])
        self.state_high = np.array(list(dyn_config_["bound"]["max"]) + [1,1,1,1,1,1.5])
        self.fext_low = np.array([-200, -200, -200, -20, -20, -18 ])
        self.fext_high = np.array([200, 200, 200, 20, 20, 18])
        self.goal_low = np.array(self.end_pos) - np.array(dyn_config_["bound"]["max"])
        self.goal_high = np.array(self.end_pos) - np.array(dyn_config_["bound"]["min"])

        self.obs_low = np.array(list(self.state_low) + list(self.fext_low) + list(self.state_low[:6]) + list(self.state_low[:6]) + list(self.state_low[:6]) + list(self.goal_low))
        self.obs_high = np.array(list(self.state_high) + list(self.fext_high) + list(self.state_high[:6]) + list(self.state_high[:6]) + list(self.state_high[:6]) + list(self.goal_high))
                
        self.observation_space = Box(low=-1*np.ones([42]),
                                high=1*np.ones([42]),
                                dtype=np.float32)
        
        self.traj_hist = []
        self.action_hist = []
        self.mjc = []
        self.dyn = DynamicsCpp.Dynamics()
        self.tree = Tree("/home/dongwon/research/configuration_learning_nut/graph_data/graph_GraphML.xml", dyn_config_)
        self.dyn_config = dyn_config_
        self.acc_cost = 0
        self.prior_state = np.array([0,0,0,0,0,0])

    def state_clip(self):
        epsilone = 0.0001*np.ones([12])
        
        if (np.sum(self.state <= self.state_low) + np.sum(self.state >= self.state_high) != 0):
            self.state[np.where(self.state <= self.state_low)] = (self.state_low[np.where(self.state <= self.state_low)]
                                                                   + epsilone[np.where(self.state <= self.state_low)])
            self.state[np.where(self.state >= self.state_high)] = (self.state_high[np.where(self.state >= self.state_high)]
                                                                   - epsilone[np.where(self.state >= self.state_high)])
            return True
        return False
        
    def get_obs(self):
        obs = np.array( list(self.state) + list(self.fext) + list(self.ref_poss) + list(self.end_pos - np.array(self.state)[:6]) )
        obs_normalized = (obs - (self.obs_low + self.obs_high)/2)*2/( self.obs_high - self.obs_low )
        
        return obs_normalized

    def traj_clip(self, traj):
        for i in range(len(list(traj))):
            traj[i] = np.clip(traj[i], self.state_low[:6], self.state_high[:6])
        return traj
        
    def reset(self):
        reset_pos = util.initial_pos_distribution()
        reset_vel = np.array([0,0,0,0,0,0])
        self.state = np.array(list(reset_pos) + list(reset_vel))
        self.state_clip()
        self.fext = np.zeros((6))
        ref_u = self.tree.LQT(self.state)
        self.ref_poss = np.array(self.traj_clip(self.tree.get_ref_traj()[:3])).flatten()
        self.goal_wrt_pos = self.end_pos - self.state[:6]
        self.nstep = 0
        self.acc_cost = 0
        
        if len(self.traj_hist) > 2000:
            self.traj_hist = self.traj_hist[:2000]
            self.action_hist = self.action_hist[:2000]
        
        return self.get_obs()

    def action_decode(self, action_normalized):
        return np.array(action_normalized)/2 * (self.action_high - self.action_low) + (self.action_high + self.action_low)/2.0

    def step(self, action):
        
        action = np.array(action)
        action_realized = self.action_decode(action)
        FF_input = np.array(action_realized[7:13])
        
        # take action
        self.tree.set_param(action_realized)
        
        for _ in range(self.dyn_config["high_level_step"]):
            ref_u = self.tree.LQT(self.state)
            for _ in range(self.dyn_config["substep"]):
                self.prior_state = self.state
                self.state = self.dyn.fwd_dynamics_holonomic(self.state, ref_u + FF_input)
                self.state_clip()
                self.acc_cost += LA.norm(np.array(self.state) - np.array(self.prior_state))
                
            self.traj_hist.append(self.state)
            self.action_hist.append(action_realized)
        self.fext = np.clip(self.dyn.get_fext(), self.fext_low, self.fext_high)
        self.ref_poss = np.array(self.traj_clip(self.tree.get_ref_traj()[:3])).flatten()
        
        done = False
        reward = 0
        self.nstep += 1
        if self.nstep >= 100 :
            print("time out done")
            reward = -0.5 - 2*self.acc_cost/1000
            done = True

        if (LA.norm(self.state[:6]-self.dyn_config['goal']) <  0.030):
            print("goal in done step : {}".format(self.nstep))
            reward = 2 - 2*self.acc_cost/1000
            done = True
                
        return self.get_obs(), reward, done, {"acc cost" : 2*self.acc_cost/100}

    def visualization_setting(self):
        self.mjc = MjcModule('/home/dongwon/research/configuration_learning_nut/xmls/bolt_nut.xml',dyn_config_)

    def visualize_traj(self):
        if self.mjc != [] :
            for j in range(len(self.traj_hist)):
                self.mjc.vis_cur_pos(self.traj_hist[j])
                time.sleep(self.dyn_config['dt'] * self.dyn_config['substep'])
            
    def traj_reset(self):
        self.traj_hist = []
        self.action_hist = []