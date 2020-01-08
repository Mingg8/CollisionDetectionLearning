import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree, BallTree
from sklearn.neighbors import DistanceMetric
from numpy import linalg as LA
import module.util as util
from scipy.spatial.transform import Rotation as sR
from scipy import linalg
from copy import deepcopy

class Tree(object):
    def __init__(self, dir, config):
        self.config = config
        self.G = nx.readwrite.graphml.read_graphml(dir)
        print("num nodes : {}".format(len(list(self.G.nodes))))
        self.weight = np.array([1,1,1,0.01,0.01,0.01])
        vers = [self.extract_coord_from_nodes(n)*self.weight for n in list(self.G.nodes)]
        # self.kdtree = KDTree(np.array(vers),metric='euclidean')
        self.kdtree = BallTree(np.array(vers),metric='euclidean')
        
        self.purturb_step = 1
        m = config["m"]
        J = config["J"]
        self.M = np.vstack((
            np.hstack(( m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , J*np.eye(3) ))
        ))
        self.invM = np.vstack((
            np.hstack(( 1/m*np.eye(3) , np.zeros((3,3)) )) ,
            np.hstack(( np.zeros((3,3)) , 1/J*np.eye(3) ))
        ))
           
        # translation + rotation        
        self.P = self.config["LQT"]["P"]
        self.Q = self.config["LQT"]["Q"]
        # R = self.config["LQT"]["R"][:3,:3]
        self.R = self.config["LQT"]["R"]
        
        self.nhorizon = self.config["LQT"]["nhorizon"] ##  cur(*) @ @ @ => 3 nhorizon
        self.ninterval = self.config["LQT"]["interval"]
        self.node_vel = self.config["LQT"]["vel_per_nodes"]
        
        
        ## A B calculation for near next state
        self.A_substep = np.vstack((
            np.hstack(( np.eye(3) , np.zeros((3,3)) , self.config["dt"] * np.eye(3) , np.zeros((3,3)) )),
            np.hstack(( np.zeros((3,3)), np.eye(3) , np.zeros((3,3)) , self.config["dt"] * np.eye(3) )),
            np.hstack(( np.zeros((3,3)), np.zeros((3,3)) , np.eye(3) , np.zeros((3,3)) )),
            np.hstack(( np.zeros((3,3)), np.zeros((3,3)) , np.zeros((3,3)) , np.eye(3) ))
        ))
        self.A = self.A_substep**self.config["substep"]

        

        
    def state_weight(self, state):
        return state*self.weight
        
    def connect_to_root(self, target):
        return nx.algorithms.shortest_paths.generic.shortest_path(self.G, source = 'n1', target=target)

    def extract_coord_from_nodes(self, node):
        return [float(i) for i in self.G.nodes[node]['coords'].split(',')]

    def get_coords_from_node_list(self, node_list):
        return [self.extract_coord_from_nodes(n) for n in node_list]
    
    def get_coords_from_init_node(self, init_node):
        return self.get_coords_from_node_list(self.connect_to_root(init_node))
    
    def find_nn(self,pos): # weighting pos here!
        wpos = np.array(pos) * self.weight
        if len(wpos.shape) <= 1:
            wpos = np.expand_dims(wpos,axis=0)
        return self.kdtree.query(wpos ,k=1)
    
    def get_nearest_path(self, pos): #non-weighted input pos
        dist, idx = self.find_nn(pos)
        node = 'n'+str(idx[0][0])
        self.nn_node = node
        nn_path = self.get_coords_from_init_node(node)
        nn_path = list(nn_path)        
        
        nn_state = self.extract_coord_from_nodes(node)
        
        ## add states
        resolution = 0.0005
        split_no = int(dist/resolution)
        for i in range(split_no):
            nn_path.append(pos + (nn_state - pos) * (split_no-i) / split_no)
        
        # nn_path.append(pos)
        
        nn_path = np.array(nn_path)
        return np.flip(nn_path, 0)
    
    
    
    def PID_controller(self, des_pos, des_vel):
        
        pos = self.state[:6]
        vel = self.state[6:]
        
        # K = np.diag([4000,4000,4000,200,200,200])
        # K = np.diag([9000,9000,9000,600,600,600])
        # K = np.diag([2000,2000,2000,100,100,100])
        K = self.config["tree_controller"]["stiffness"]
        B = 2*np.sqrt(self.M*K)
        
        error_tran = pos[:3] - des_pos[:3]
        error_rot = util.XYZEuler2R(pos[3],pos[4],pos[5]) @ np.transpose(util.XYZEuler2R(des_pos[3],des_pos[4],des_pos[5]))
        error_rot_vec = sR.from_dcm(error_rot).as_rotvec()
        
        error = np.array(list(error_tran) + list(error_rot_vec))
        des_vel_rot_zero = np.zeros((6))
        des_vel_rot_zero[:3] = des_vel[:3]
        
        return -K@error - B@(vel - util.inv_Jacoxs(self.state)@des_vel)

    def BdynCal(self, pos_des):
        
        iMxx = self.invM[:3,:3]
        iMrr = self.invM[3:,3:]
        
        B_substep = np.vstack((
            np.hstack(( self.config["dt"]**2/2*iMxx , np.zeros((3,3)) )),
            np.hstack(( np.zeros((3,3)) , self.config["dt"]**2/2 * util.inv_euler_jaco(pos_des[3:6]) @ iMrr )),
            np.hstack(( self.config["dt"]*iMxx , np.zeros((3,3)) )),
            np.hstack(( np.zeros((3,3)) , self.config["dt"] * util.inv_euler_jaco(pos_des[3:6]) @ iMrr ))
        ))
        
        rec = np.identity(12)
        for i in range(self.config["substep"]):
            rec = self.A_substep@rec + np.identity(12)
        B = rec @ B_substep

        return B


    def LQT(self, state):
        """
        problem is..
        finite horison tracking.. discrete version
        """

        # control_dt = (self.config["dt"] * self.config["substep"])
        # control_dt = self.config["LQT"]["control_dt"]
        
        # iMxx = self.invM[:3,:3]
        # iMrr = self.invM[3:,3:]
        
        # translation + rotation        
        P = self.P
        Q = self.Q
        R = self.R
        
        nhorizon = self.nhorizon
        ninterval = self.ninterval

        pos = state[:6]
        # tran_state = state[([0,1,2,6,7,8])]
        # ang_state = state[([3,4,5,9,10,11])]
        remaining_path = self.get_nearest_path(pos)
            
        # padding goal position when near goal
        if len(remaining_path) <= nhorizon*ninterval :
            remaining_path = list(remaining_path)
            for i in range(nhorizon*ninterval) :
                remaining_path += [remaining_path[-1]]
            remaining_path = np.array(remaining_path)                                      
                    
        self.ref_traj = remaining_path
                    
        pos_des = []
        for i in range(nhorizon):
            pos_t = remaining_path[ninterval*(i)]
            pos_nt = remaining_path[ninterval*(i+1)]
            vel_tmp = self.node_vel * (pos_nt - pos_t)/ ninterval
            # des_tmp = np.array(list(pos_nt) + list(vel_tmp))
            des_tmp = np.array(list(pos_t) + list(vel_tmp))
            pos_des.append(des_tmp)  
        
        ## A B calculation for near next state
        # A_substep = np.vstack((
        #     np.hstack(( np.eye(3) , np.zeros((3,3)) , self.config["dt"] * np.eye(3) , np.zeros((3,3)) )),
        #     np.hstack(( np.zeros((3,3)), np.eye(3) , np.zeros((3,3)) , self.config["dt"] * np.eye(3) )),
        #     np.hstack(( np.zeros((3,3)), np.zeros((3,3)) , np.eye(3) , np.zeros((3,3)) )),
        #     np.hstack(( np.zeros((3,3)), np.zeros((3,3)) , np.zeros((3,3)) , np.eye(3) ))
        # ))
        # B_substep = np.vstack((
        #     np.hstack(( self.config["dt"]**2/2*iMxx , np.zeros((3,3)) )),
        #     np.hstack(( np.zeros((3,3)) , self.config["dt"]**2/2 * util.inv_euler_jaco(pos_des[0][3:6]) @ iMrr )),
        #     np.hstack(( self.config["dt"]*iMxx , np.zeros((3,3)) )),
        #     np.hstack(( np.zeros((3,3)) , self.config["dt"] * util.inv_euler_jaco(pos_des[0][3:6]) @ iMrr ))
        # ))
        
        # A = A_substep**self.config["substep"]
        # rec = np.identity(12)
        # for i in range(self.config["substep"]):
        #     rec = A_substep@rec + np.identity(12)
        # B = rec @ B_substep
        
        
        
        # if len(remaining_path) == 1 or len(remaining_path) == 2:
        #     K = [-linalg.inv(R+B.T@P@B)@B.T@P@A]
        #     kv = [linalg.inv(R+B.T@P@B)@B.T@P@pos_des[-1]]
        # else :
        Vxx = [None]*nhorizon
        Vx = [None]*nhorizon
        Quu = [None]*(nhorizon-1)
        Qux = [None]*(nhorizon-1)
        Qxx = [None]*(nhorizon-1)
        Qu = [None]*(nhorizon-1)
        Qx = [None]*(nhorizon-1)
        K = [None]*(nhorizon-1)
        kv = [None]*(nhorizon-1)
        
        Vxx[-1] = P
        Vx[-1] = -P@pos_des[-1]
        
        for k in range((nhorizon-1)-1,-1,-1): # k : nhorizon-1-1 --> 0
            # A, B calculation
            
            B = self.BdynCal(pos_des[k])
            
            # 1 conversion
            Qxx[k] = Q + self.A.T@Vxx[k+1]@self.A
            Qux[k] = B.T@Vxx[k+1]@self.A
            Quu[k] = B.T@Vxx[k+1]@B + R
            Qx[k] = self.A.T@Vx[k+1] - Q@pos_des[k]
            Qu[k] = B.T@Vx[k+1]
            
            # 3 conversion
            K[k] = -linalg.inv(Quu[k])@Qux[k]
            kv[k] = -linalg.inv(Quu[k])@Qu[k]
            
            # 4 conversion
            Vxx[k] = K[k].T@Quu[k]@K[k] + K[k].T@Qux[k] + Qux[k].T@K[k] + Qxx[k]
            Vx[k] = K[k].T@Quu[k]@kv[k] + Qux[k].T@kv[k] + K[k].T@Qu[k] + Qx[k]
    
        return K[0]@state + kv[0]


    def tree_follower(self, state):
        self.state = np.array(state)
        pos = self.state[:6]
        vel = self.state[6:]
        
        rs = self.config["tree_controller"]["receding_step"]
        prs = self.config["tree_controller"]["purturbed_receding_step"]
        
        remaining_path = self.get_nearest_path(pos)
        if len(remaining_path) <= prs+1:
            des_pos = remaining_path[-1]
        elif LA.norm( vel*np.array([1,1,1,0.01,0.01,0.01]) ) > 0.004 and self.purturb_step == 0:
            des_pos = remaining_path[rs]
            
        # prevent stuck
        elif self.purturb_step == 0:
            self.purturb_step = int(0.10/self.config["dt"])+1
            # print("inside here!? 1st")
            des_pos = remaining_path[prs]
        else :
            # print("inside here!? 2nd {} // state : {}".format(self.purturb_step, pos))
            self.purturb_step = self.purturb_step -1
            des_pos = remaining_path[prs]

        # des_vel = (des_pos - pos) / (self.config["dt"] * self.config["substep"]) 
        des_vel = self.config["des_following_vel"] * (des_pos - pos) / self.config["tree_controller"]["receding_step"] # step / s
        
        return self.PID_controller(des_pos, des_vel)
        
    def set_param(self, action):
        action = np.array(action)
        
        Q_tran = action[0]
        Q_e = action[1]
        Q_v = action[2]
        Q_de = action[3]
        
        self.Q = np.diag([Q_tran,Q_tran,Q_tran,Q_e,Q_e,Q_e,Q_v,Q_v,Q_v,Q_de,Q_de,Q_de])
        self.P = deepcopy(self.Q)
        
        self.ninterval = int(action[4])
        self.node_vel = action[5]
        self.nhorizon = int(action[6])
        
    def get_ref_traj(self):
        return self.ref_traj
        

if __name__ == "__main__":

    tree = Tree("data_save/graph_GraphML.xml")
    # print(tree.get_coords_from_init_state('n100'))
    # dist, idx = tree.find_nn(np.array([0,0,0,0,0,0]))
    print(tree.get_nearest_path(np.array([0,0,0,0,0,0])))