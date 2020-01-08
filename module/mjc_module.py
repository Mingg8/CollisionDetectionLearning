from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import normalize
from mujoco_py import MjSim, MjViewer, MjViewerBasic, load_model_from_path
import glfw
from sklearn.utils import shuffle
import math
import time
from mujoco_py.generated import const
# import imageio
# from multiprocessing import Process, Queue
from scipy.spatial.transform import Rotation as sR

class DWMjViewer(MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        glfw.set_key_callback(self.window, self.key_callback_DW)
        self._hide_overlay = True
        self.key_press = "none"
        self.cam.fixedcamid += 1
        self.cam.type = const.CAMERA_FIXED
        # self.cam.type = const.CAMERA_FIXED
        
    def key_callback_DW(self, window, key, scancode, action, mods):
        self.key_press = key
        if action == glfw.RELEASE and key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            exit(0)
        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_TAB:  # Switches cameras.
            self.cam.fixedcamid += 1
            self.cam.type = const.CAMERA_FIXED
            if self.cam.fixedcamid >= self._ncam:
                self.cam.fixedcamid = -1
                self.cam.type = const.CAMERA_FREE
    


class MjcModule(object):
    def __init__(self, xml_dir, config, dt = 0.001, nstep = 1000):
        self.dt = dt
        self.nstep = nstep
        
        # self.model = load_model_from_path("xmls/peg_2d_real_exp_large_gap.xml")
        self.model = load_model_from_path(xml_dir)
        self.sim = MjSim(self.model)
        self.d = self.sim.data
        self.viewer = DWMjViewer(self.sim)
        

        
        self.m = self.model.body_mass[1]
        self.J = self.model.body_inertia[1][0]
        
        glfw.set_window_pos(self.viewer.window,1300,100)
        glfw.set_window_size(self.viewer.window,700,800)
    
    def vis_cur_pos(self, pos):
        for i in range(6):
            self.d.qpos[i] = pos[i]
            
        self.sim.forward()
        self.viewer.render()

    def set_init(self):
        if np.random.rand(1) < 0.10:
            ini_qpos = np.array([0.0, -0.020-0.010*np.random.rand(1), 0.0])
            ini_qvel = np.array([0, 0, 0])
        else :            
            ini_qpos = np.array([0.010*np.random.randn(1), 0.020 + 0.005*np.random.randn(1), 0.5*np.random.randn(1)])
            ini_qvel = 0.03*np.array([0.1*np.random.randn(1), 0.05*np.random.randn(1), 0.1*np.random.randn(1)])
            
        self.sim.reset()
        for i in range(len(ini_qpos)):
            self.d.qpos[i] = ini_qpos[i]
            self.d.qvel[i] = ini_qvel[i]
        self.sim.step()
        
    def draw_arrow(self, body_idx1, body_idx2, s1, s2):
        vec = np.squeeze(normalize([s2-s1]))
        self.model.body_pos[body_idx1] = np.array(s1)
        self.model.body_pos[body_idx2] = np.array(s2)
        # rot_axis = np.cross(np.array([0,0,1]),vec)
        # rot_angle = math.atan2(LA.norm(rot_axis), np.inner(vec,np.array([0,0,1])))
        # quat_from_sR = sR.from_rotvec(rot_angle*rot_axis).as_quat()    
        # self.model.body_quat[body_idx2] = np.array([quat_from_sR[3], quat_from_sR[0], quat_from_sR[1], quat_from_sR[2]])
    
        z_axis = vec
        x_axis = np.squeeze(normalize([np.cross(np.array([1,0,0]),vec)]))
        y_axis = np.squeeze(normalize([np.cross(vec , x_axis)]))
        
        R = np.transpose(np.array(list([x_axis]) + list([y_axis]) + list([z_axis])))
        
        quat_from_sR = sR.from_dcm(R).as_quat()
        self.model.body_quat[body_idx2] = np.array([quat_from_sR[3], quat_from_sR[0], quat_from_sR[1], quat_from_sR[2]])
    
    
    def draw_real_bolt(self, body_idx, state):
        self.model.body_pos[body_idx] = np.array(state[:3])
        quat_from_sR = sR.from_euler("xyz",state[3:]).as_quat()
        self.model.body_quat[body_idx] = np.array([quat_from_sR[3], quat_from_sR[0], quat_from_sR[1], quat_from_sR[2]])