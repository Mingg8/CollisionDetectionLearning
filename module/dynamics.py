from __future__ import absolute_import, division, print_function, unicode_literals
from collections import deque
import numpy as np
import numpy.linalg as LA
import scipy.signal
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import module.util as util
from config import config

class Dynamics(object):
    def __init__(self, mlp):
        self.m = config["m"]
        self.J = config["J"]
        self.dt = config["dt"]
        self.stabilization_coeff = config["stabilization_coeff"]
        self.damping_coeff = config["damping_coeff"]
        self.threshold = config["threshold"]
        self.bound = config["bound"]       
        self.mlp = mlp
        self.M = np.vstack((
            np.hstack(( self.m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , self.J*np.eye(3) ))
        ))
        
    def fwd_dynamics_nonsmooth(self, state, u):
        
        pos = state[:6]
        vel = state[6:]
        
        pos_ex = np.expand_dims(pos,axis=0)
        
        M = np.vstack((
            np.hstack(( self.m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , self.J*np.eye(3) ))
        ))
        self.M = M
        invM = np.vstack((
            np.hstack(( 1/self.m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , 1/self.J*np.eye(3) ))
        ))
        
        # A = self.mlp.grad_c_cal(pos_ex)
        epsilone = self.mlp.predict(pos_ex)
        # epsilone = pos[2]
        if epsilone >= self.threshold:
            lambda_x = np.zeros(6)
        else:
            J = util.Jacoxs(state)
            # A = self.mlp.grad(pos_ex)
            A = normalize(self.mlp.grad(pos_ex)[:,np.newaxis], axis=0).ravel()
            # A = np.array([0,0,1,0,0,0])
        
            AJ = A @ J
            D = AJ @ invM @ np.transpose(AJ)
            b = A@vel + AJ @ invM @ u * self.dt
            if b >= 0:
                lambda_x = np.zeros(6)
            else:
                damping = - A@vel * self.damping_coeff
                b_des = damping -self.stabilization_coeff / self.dt * (epsilone -self.threshold)- b
                lambda_ = 1/D*b_des
                lambda_x = np.transpose(AJ) * lambda_
        
        vel_nxt = vel + self.dt*invM @ u + invM@lambda_x
        vel_hat = (vel + vel_nxt)/2.0
        pos_nxt = pos + vel_hat*self.dt
        
        # if LA.norm(vel_nxt) > 50:
            # print("error? : {}".format(vel_nxt))
        
        return np.array(list(pos_nxt) + list(vel_nxt))
    
    def fwd_dynamics_wo_contact(self, state, u):
        
        pos = state[:6]
        vel = state[6:]
        
        pos_ex = np.expand_dims(pos,axis=0)
        
        M = np.vstack((
            np.hstack(( self.m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , self.J*np.eye(3) ))
        ))
        self.M = M
        invM = np.vstack((
            np.hstack(( 1/self.m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , 1/self.J*np.eye(3) ))
        ))
                
        vel_nxt = vel + self.dt*invM @ u
        vel_hat = (vel + vel_nxt)/2.0
        pos_nxt = pos + vel_hat*self.dt
        
        # if LA.norm(vel_nxt) > 10:
            # print("error? : {}".format(vel_nxt))
        
        return np.array(list(pos_nxt) + list(vel_nxt))
    
    
    def compound_fwd_dynamics(self, state, u):
        pos = state[:6]
        if ( np.where( (pos <= config["no_contact_bound"]['max']) == True )[0].size == 6 and
            np.where( (pos >= config["no_contact_bound"]['min']) == True )[0].size == 6 ):
            state = self.fwd_dynamics_wo_contact(state,u)
            # print("integrate here! state : {}".format(list(pos)))
        else:
            state = self.fwd_dynamics_nonsmooth(state,u)
            
        return state