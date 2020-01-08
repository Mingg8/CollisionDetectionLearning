import numpy as np
from module.gym.ezpickle import EzPickle
from module.gym import mujoco_env
import mujoco_py
import time

class ReacherEnv(mujoco_env.MujocoEnv):
    def __init__(self, config):
        # EzPickle.__init__(self)
        self.traj_hist = []
        self.substep = 0
        mujoco_env.MujocoEnv.__init__(self, '/home/dongwon/research/configuration_learning_nut/xmls/reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        self.traj_hist.append(ob)
        
        if self.substep > 100:
            done = True
        
        self.substep += 1
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.substep = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def visualization_setting(self):
        self.viewer = mujoco_py.MjViewer(self.sim)

    def visualize_traj(self):
        if (len(self.traj_hist) > 350):
            for j in range(len(self.traj_hist)-350,len(self.traj_hist)):
                for i in range(2):
                    self.sim.data.qpos[i] = self.traj_hist[j][i]
                self.sim.step()
                self.viewer.render()
                time.sleep(self.model.opt.timestep)
        else:
            for j in range(len(self.traj_hist)):
                for i in range(2):
                    self.sim.data.qpos[i] = self.traj_hist[j][i]
                self.sim.step()
                self.viewer.render()
                time.sleep(self.model.opt.timestep)
            
    def traj_reset(self):
        self.reset()
        self.traj_hist = []