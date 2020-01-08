import os, sys
sys.path.insert(0, os.path.abspath("/home/dongwon/research/configuration_learning_nut"))
import numpy as np
import numpy.linalg as LA
from config import config
import random
from scipy.spatial.transform import Rotation as sR
import math
from math import sin, cos, tan, asin, acos, atan2

def XYZEuler2R(x,y,z):
    res = sR.from_rotvec(x * np.array([1, 0, 0])) * sR.from_rotvec(y * np.array([0, 1, 0])) * sR.from_rotvec(z * np.array([0, 0, 1]))
    return res.as_dcm()

def R2XYZEuler(iR):
    y = asin(-iR[0,2])
    x = atan2(-iR[1,2]/cos(y),iR[2,2]/cos(y))
    z = atan2(-iR[0,1]/cos(y),iR[0,0]/cos(y))
    return np.array([x,y,z])

def R_diff(R1,R2):
    return LA.norm(sR.from_dcm(np.transpose(R2)@R1).as_rotvec())


def collision_free_config_gen(angle, output_type = 0):
    tran = np.array([0, 0, 0.0461-0.005*(angle)/(-2*3.14159265358)])
    if output_type == 0:
        return tran, sR.from_rotvec(angle * np.array([0, 0, 1])).as_dcm()
    elif output_type ==1:
        return np.array((list(tran) + [0,0,angle]))

def random_state_generation(num):
    bound_ = config["bound"]
    state = np.squeeze(np.random.random(size=[num,6]))
    # state = np.array([random.random(), random.random(),random.random(),random.random(),random.random(),random.random()])
    state = state - 0.5*np.ones([6])
    state = state * (bound_["max"] - bound_["min"])
    state = state + (bound_["max"] + bound_["min"])/2.0
    return state


def random_states_generation(num_states):
    bound_ = config["bound"]
    states = np.random.rand(num_states,6)
    states = states - 0.5*np.ones([num_states,6])
    states = states * (bound_["max"] - bound_["min"])
    states = states + (bound_["max"] + bound_["min"])/2
    
    return states

def euler_jaco(s):
    return np.array([ [1 , 0, sin(s[1]) ],
                     [ 0, cos(s[0]), -cos(s[1])*sin(s[0])  ],
                     [ 0, sin(s[0]), cos(s[0])*cos(s[1])] ])
    
def inv_euler_jaco(s):  # de = inv_euler_jaco*w  (de : XYZEuler vel / w : angular vel)
    x = s[0]
    y = s[1]
    z = s[2]
    return np.array([ [1 , sin(x)*sin(y)/cos(y), -cos(x)*sin(y)/cos(y) ],
                     [ 0, cos(x), sin(x)  ],
                     [ 0, -sin(x)/cos(y), cos(x)/cos(y)] ])

def Jacoxs(s):   # ds = Jacoxs*dx  (x : SE3 / s : R3XYZEuler)
    return np.vstack((
        np.hstack((np.identity(3) , np.zeros((3,3)) )),
        np.hstack((np.zeros((3,3)) , inv_euler_jaco(s[3:]) ))
    ))

def inv_Jacoxs(s):   # dx = inv_Jacoxs*ds
    return np.vstack((
        np.hstack((np.identity(3) , np.zeros((3,3)) )),
        np.hstack((np.zeros((3,3)) , euler_jaco(s[3:]) ))
    ))

def initial_pos_distribution():
    state = np.random.normal(0, 0.004, [6]) * np.array([1,1,1,30,30,100]) + np.array([0,0,0.060,0,0,0])
    return state

    
if __name__ =="__main__":
    print(random_states_generation(100))
