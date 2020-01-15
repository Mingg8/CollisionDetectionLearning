from multiprocessing import Pool
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as sR
import random
import pickle
from config import config
import module.util as util
from module.evaluation import plot_R3_space_pred, plot_R3_space_input, plot_near_CF_path_pred, plot_near_CF_path_fcl
from copy import deepcopy
import DynamicsCpp
# from module.mjc_module import MjcModule
from module.tree import Tree
from config import config
import numpy.linalg as LA
import functools
from multiprocessing.managers import BaseManager

import multiprocessing
multiprocessing.set_start_method('spawn', True)

from sklearn.utils import shuffle
import module.MLP as MLP

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    # vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(int(string[i:-1].split("/")[0])-1)
                            break
                        face.append(int(string[i:string.find(" ", i)].split("/")[0])-1)
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            self.vertices = np.array(self.vertices).astype(float)
            self.faces = np.array(self.faces).astype(int)
            
            f.close()
        except IOError:
            print(".obj file not found.")


class CollisionChecker(object):
    def __init__(self):
        
        env_mesh = ObjLoader("obj/coarse_bolt_v3_size_reduced.obj")
        obj_mesh = ObjLoader("obj/nut.obj")

        self.m1 = fcl.BVHModel()
        self.m1.beginModel(len(env_mesh.vertices), len(env_mesh.faces))
        self.m1.addSubModel(env_mesh.vertices, env_mesh.faces)
        self.m1.endModel()
            
        self.m2 = fcl.BVHModel()
        self.m2.beginModel(len(obj_mesh.vertices), len(obj_mesh.faces))
        self.m2.addSubModel(obj_mesh.vertices, obj_mesh.faces)
        self.m2.endModel()
        
        R_init = np.array([[1.0, 0.0, 0.0],
                    [0.0,  1.0, 0.0],
                    [0.0,  0.0, 1.0]])
        T_init = np.array([0.0, 0.0, 0.0])

        t = fcl.Transform(R_init, T_init)
        self.obj1 = fcl.CollisionObject(self.m1, t)
        self.obj2 = fcl.CollisionObject(self.m2, t)
    
    def collisioncheck(self, rot, tran):
        r1 = util.XYZEuler2R(0.0, 0.0, 0.0)
        tran1 = np.array([0.0, 0.0, 0.0])
        t1 = fcl.Transform(r1, tran1)
        t2 = fcl.Transform(rot, tran)
        self.obj1.setTransform(t1)   # Using a transform
        self.obj2.setTransform(t2)   # Using a transform
        request = fcl.CollisionRequest(enable_contact=True)
        result = fcl.CollisionResult()
        ret = fcl.collide(self.obj1, self.obj2, request, result)
        
        if result.contacts == []:
            return int(~(result.is_collision)) + 1.5
        else :
            pen_depth = result.contacts[0].penetration_depth
            max_pene_depth = config["max_penetration_depth"] # 1.5 mm
            return -0.4-np.clip(result.contacts[0].penetration_depth, 0, max_pene_depth)/max_pene_depth/2
    
    def collisioncheck_from_state(self, state):
        tran = state[:3]
        rot = util.XYZEuler2R(state[3],state[4],state[5])
        return self.collisioncheck(rot,tran)

# class CDManager(BaseManager):  
#     pass
# CDManager.register('CollisionChecker', CollisionChecker, exposed = ['collisioncheck','collisioncheck_from_state'])

class DataGeneration(object):
    def __init__(self, id):
        self.iter_num = 1000
        self.id = id
        self.CD = CollisionChecker()
        
        # env_mesh = ObjLoader("obj/coarse_bolt_v3_size_reduced.obj")
        # obj_mesh = ObjLoader("obj/nut.obj")

        # self.m1 = fcl.BVHModel()
        # self.m1.beginModel(len(env_mesh.vertices), len(env_mesh.faces))
        # self.m1.addSubModel(env_mesh.vertices, env_mesh.faces)
        # self.m1.endModel()
            
        # self.m2 = fcl.BVHModel()
        # self.m2.beginModel(len(obj_mesh.vertices), len(obj_mesh.faces))
        # self.m2.addSubModel(obj_mesh.vertices, obj_mesh.faces)
        # self.m2.endModel()

        self.bound = {}
        self.bound["max"] = np.array([0.030,  0.030,  0.070,  1.00,  1.00,  3.15])
        self.bound["min"] = np.array([-0.030, -0.030,  0.000, -1.00, -1.00, -11.00])

        self.LIOs = {}
        self.LIOs["state"] = []
        self.LIOs["result"] = []
        
    #     R_init = np.array([[1.0, 0.0, 0.0],
    #                 [0.0,  1.0, 0.0],
    #                 [0.0,  0.0, 1.0]])
    #     T_init = np.array([0.0, 0.0, 0.0])

    #     t = fcl.Transform(R_init, T_init)
    #     self.obj1 = fcl.CollisionObject(self.m1, t)
    #     self.obj2 = fcl.CollisionObject(self.m2, t)
    
    # def collisioncheck(self, rot, tran):
    #     r1 = util.XYZEuler2R(0.0, 0.0, 0.0)
    #     tran1 = np.array([0.0, 0.0, 0.0])
    #     t1 = fcl.Transform(r1, tran1)
    #     t2 = fcl.Transform(rot, tran)
    #     self.obj1.setTransform(t1)   # Using a transform
    #     self.obj2.setTransform(t2)   # Using a transform
    #     request = fcl.CollisionRequest(enable_contact=True)
    #     result = fcl.CollisionResult()
        
    #     ret = fcl.collide(self.obj1, self.obj2, request, result)
        
    #     if result.contacts == []:
    #         return int(~(result.is_collision)) + 1.5
    #     else :
    #         pen_depth = result.contacts[0].penetration_depth
    #         max_pene_depth = config["max_penetration_depth"] # 1.5 mm
    #         return -0.4-np.clip(result.contacts[0].penetration_depth, 0, max_pene_depth)/max_pene_depth/2
    
    # def collisioncheck_from_state(self, state):
    #     tran = state[:3]
    #     rot = util.XYZEuler2R(state[3],state[4],state[5])
    #     return self.collisioncheck(rot,tran)
    
    def gen(self):
        for i in range(self.iter_num):
            state = util.random_state_generation(1)
            # state[3:] = np.zeros([3])
            
            r2 = util.XYZEuler2R(state[3], state[4], state[5])
            tran2 = state[:3]
            
            self.LIOs["state"].append(state)
            # self.LIOs["result"].append(int(self.collisioncheck(r2,tran2))+1.5)
            self.LIOs["result"].append(self.CD.collisioncheck(r2,tran2))
        return self.LIOs
        
    def gen_near_state(self, state):
        # noise_weight = np.array([0.001, 0.001, 0.001, 0.03, 0.03, 0.06])
        noise_weight = np.array([0.0003, 0.0003, 0.0002, 0.005, 0.005, 0.01])
        # noise_weight = np.zeros([6])
        noise = np.random.normal(0, 1, [70,6]) * noise_weight
        noise = np.vstack((noise , np.zeros([1,6]))) # for original data
    
        arg_state = noise + state
        arg_state = list(arg_state)
        
        labels = []
        for i in range(len(arg_state)):
            labels.append(self.CD.collisioncheck_from_state(arg_state[i]))
        
        labels = np.array(labels)
        if len(list(labels[np.where(labels == 0.5)])) == 0:
            arg_state = [state]
            labels = [labels[-1]]
        elif len(list(labels[np.where(labels < 0)])) == 0:
            arg_state = [state]
            labels = [labels[-1]]
        
        return list(arg_state), list(labels)
        
def data_gen_for_par(id):
    datagen =DataGeneration(id)
    return datagen.gen()

def data_gen_near_demonstration_for_par(state):
    datagen =DataGeneration(0)
    return datagen.gen_near_state(state)

def test_collision_validity(): # compared with nominal collision free trajectory
    datagen =DataGeneration(0)
    resolution = 1000
    for i in range(resolution):
        tran, rot = util.collision_free_config_gen(-i/resolution*4*np.pi)
        # tran[2] += 0.00016 #  # max boundary of contact
        # tran[2] -= 0.000115 # min boundary of contact
        # print("collision? : {}, state : {}, rot : \n{}".format(datagen.collisioncheck(rot,tran),tran, rot))
        print("collision? : {}, state : {}".format(datagen.collisioncheck(rot,tran),tran))

def classes_count(LIO):
    output = np.array(LIO['output'])
    input_ = np.array(LIO['input'])
    idx = np.where(output==0.5)
    out_idx = np.where(output < 0)
    num_total_data = len(LIO['output'])
    num_inside_data = np.shape(output[idx])[0]
    num_outsize_data1 = num_total_data - num_inside_data
    print("bump 1 - total_data {} // inside data : {} // outside data : {}".format(num_total_data, num_inside_data, num_total_data - num_inside_data))
    
    new_input = np.array(LIO['input'])[idx]
    
    deep_nut_idx = np.where(new_input[:,2] < 0.0445)
    num_total_data = len(new_input)
    num_inside_data = np.shape(input_[deep_nut_idx])[0]
    num_outside_data = num_total_data - num_inside_data
    print("bump 2 - total_data {} // inside data : {} // outside data : {}".format(num_total_data, num_inside_data, num_total_data - num_inside_data))
    
    return new_input[deep_nut_idx]



def data_info_print(LIO):
    output = np.array(LIO['output'])
    input_ = np.array(LIO['input'])
    idx = np.where(output==0.5)
    out_idx = np.where(output<0)
    num_total_data = len(LIO['output'])
    num_inside_data = np.shape(output[idx])[0]
    num_outsize_data1 = num_total_data - num_inside_data
    print("cur data - total_data {} // inside data : {} // outside data : {}".format(num_total_data, num_inside_data, num_total_data - num_inside_data))
    
    new_input = np.array(LIO['input'])[idx]
    
    deep_nut_idx = np.where(new_input[:,2] < 0.0445)
    num_total_data = len(new_input)
    num_inside_data = np.shape(input_[deep_nut_idx])[0]
    num_outside_data = num_total_data - num_inside_data
    print("valid data - total_data {} // inside data : {} // outside data : {}".format(num_total_data, num_inside_data, num_total_data - num_inside_data))


def data_balancing(LIO):
    output = np.array(LIO['output'])
    input_ = np.array(LIO['input'])
    in_idx = np.where(output==0.5)
    out_idx = np.where(output<0)
    
    input_inside = input_[in_idx]
    output_inside = output[in_idx]
    input_outside = input_[out_idx]
    output_outside = output[out_idx]
    
    num_total_data = len(LIO['output'])
    num_inside_data = np.shape(output_inside)[0]
    num_outsize_data = np.shape(output_outside)[0]
    
    print("before - total_data {} // inside data : {} // outside data : {}".format(num_total_data, num_inside_data, num_outsize_data))
    
    if (num_outsize_data > num_inside_data*1.1):
        input_outside, output_outside = shuffle(input_outside, output_outside)
        input_outside, output_outside = input_outside[:num_inside_data], output_outside[:num_inside_data]
    
    LIO['output'] = list(output_inside) + list(output_outside)
    LIO['input'] =  list(input_inside) + list(input_outside)
    
    print("after - total_data {} // inside data : {} // outside data : {}".format(len(LIO['output']), len(list(output_inside)), len(list(output_outside))))
    
    LIO['output'], LIO['input'] = shuffle(np.array(LIO['output']), np.array(LIO['input']))
    
    return LIO


def artificial_nominal_data():
    print("plot near collision free path")
    resolution = 15000
    total_angle = 4.2*np.pi
    start_i = int((140/180*np.pi)*resolution/total_angle)
    print(start_i)
    states = []
    for i in range(start_i,resolution):
        states.append(util.collision_free_config_gen(-i/resolution*total_angle, output_type=1))
    
    states += (list(np.array(states) + np.array([0,0,0.0001,0,0,0]))+
        list(np.array(states) + np.array([0,0,-0.0001,0,0,0]))+
        list(np.array(states) + np.array([0.0001,0,0,0,0,0]))+
        list(np.array(states) + np.array([-0.0001,0,0,0,0,0]))
        )
    return states


def data_rejection(LIO, mlp):
    label_pred = mlp.predict(np.array(LIO["input"]))
    input_pnts = np.array(LIO["input"])
    labels = np.array(LIO["output"])
    
    label_pred = np.squeeze(label_pred)
    rej_criteria = np.random.rand(input_pnts.shape[0])
    # prob_dis = 1.0+pred*2
    prob_dis = 0.5 + label_pred + 0.004
    outside_important_pnts_idx = np.where( (prob_dis>=rej_criteria) * (labels < 0)  )
    
    prob_dis = 0.5 - label_pred + 0.004
    inside_important_pnts_idx =  np.where( (prob_dis>=rej_criteria) * (labels == 0.5)  )
    
    new_input = list(input_pnts[outside_important_pnts_idx]) + list(input_pnts[inside_important_pnts_idx])
    new_output = list(labels[outside_important_pnts_idx]) + list(labels[inside_important_pnts_idx])

    LIO["input"] = new_input
    LIO["output"] = new_output
    
    return LIO


# def test_run(dyn, mjc, vis = True, contact = False):
#     state = np.array(list(util.initial_pos_distribution()) + [0,0,0,0,0,0])
#     pos_stack_tmp = []
#     for i in range(2500):
#         u = tree.tree_follower(state)
#         u += 0.001*40/(config["substep"] * config["dt"])  * np.random.normal(0,1,[6]) * np.array([1,1,1,0.5,0.5,0.5])
#         for i in range(config["substep"]):
#             if contact:
#                 state = dyn.fwd_dynamics(state, u)
#             else:
#                 state = dyn.fwd_dynamics_wo_contact(state, u)
#         pos_stack_tmp.append(state[:6])
#         if LA.norm(state[:6] - config["goal"]) < 0.030:
#             state[:6] = util.initial_pos_distribution()
#             break
#         if vis:
#             mjc.vis_cur_pos(state[:6])
#     return pos_stack_tmp[int(len(pos_stack_tmp)/2):]

def data_gen_from_simulation(idx):
    print("idx {} sim start".format(idx))
    dyn_in = DynamicsCpp.Dynamics()
    tree = Tree("graph_data/graph_GraphML.xml", config)
    state = np.array(list(util.initial_pos_distribution()) + [0,0,0,0,0,0])
    pos_stack_tmp = []
    for i in range(2500):
        u = tree.tree_follower(state)
        u += 0.001*33/(config["substep"] * config["dt"])  * np.random.normal(0,1,[6]) * np.array([1,1,1,0.5,0.5,0.5])
        for i in range(config["substep"]):
            state = dyn_in.fwd_dynamics_wo_contact(state, u)
        pos_stack_tmp.append(state[:6])
        if LA.norm(state[:6] - config["goal"]) < 0.030:
            state[:6] = util.initial_pos_distribution()
            break
    
    print("idx {} sim end".format(idx))

    return pos_stack_tmp[int(len(pos_stack_tmp)/2):]
    
def concentration(input, mlp):
    label_pred = mlp.predict(np.array(input))
    
    prob_dist = 2*np.absolute(label_pred) - 0.01
    
    rej_criteria = np.random.rand(len(list(input)))
    important_pnts_idx = np.where(prob_dist<=rej_criteria)
    
    return np.array(input)[important_pnts_idx]

def labeling(inputs):
    CD = CollisionChecker()
    return [[input, CD.collisioncheck_from_state(np.array(input))] for input in inputs]

if __name__ == "__main__":
    
    # initialization
    # mjc = MjcModule('xmls/bolt_nut.xml',config)
    mlp = MLP.MLP(config, hdim=64)
    dyn = DynamicsCpp.Dynamics()
    dyn.change_param(config["dt"], config["threshold"], config["damping_coeff"], config["stabilization_coeff"])
    tree = Tree("graph_data/graph_GraphML.xml", config)
    
    # learning continue! for load weight
    mlp.load_model('model_save/model0.h5')
    mlp.load_weights_to_dyn(dyn)
    # mlp.export_weight()
    bound = config["bound"]
    LIO = {}
    LIO["state"] = []    
    LIO["input"] = []
    LIO["output"] = []
        
    # data_gen_near_demonstration_for_par(np.array([0,0,0.03,0,0,0]))
    
    for itr in range(100):
        print("===== iteration {} start".format(itr))
        ## step 1 - exploration and data generation from demonstration
        print("step 1 - data generation by tree-follower")
        # data from prior iteration
        LIO["input"] = list(LIO["input"])
        LIO["output"] = list(LIO["output"])
        
        new_input = []
        with Pool(45) as p:
            generated_pos_data = p.map(data_gen_from_simulation, range(200))
            
        for i in range(len(generated_pos_data)):
            new_input += list(generated_pos_data[i])
            
        print("generated data : {}".format(len(new_input)))    
        
        ## step 2 - generate scattered data
        print("step 2 - generate scattered data start")
        scattered_data = []
        for i in range(len(list(new_input))):
            noise_weight = np.array([0.0003, 0.0003, 0.0002, 0.005, 0.005, 0.01])
            noise = np.random.normal(0, 1, [100,6]) * noise_weight
            noise = np.vstack((noise , np.zeros([1,6]))) # for original data
            scattered_data += list(noise + new_input[i])
        new_input += list(scattered_data)
        ## uniform state generation
        new_input += list(util.random_state_generation(3000000))       
        
        ## step 3 - data rejection by trained neural network
        if itr >= 0 :
            print("step 3 - data rejection from tranined data")
            print("before - num new states : {}".format(len(list(new_input))))
            new_input = concentration(new_input, mlp)
            print("after - num new states : {}".format(len(list(new_input))))
        
        ## step 4 - labeling
        print("step 4 - labeling start")
        distributed_input = []
        for i in range(int(len(new_input)/200)):
            distributed_input.append(new_input[200*i:200*(i+1)])
        with Pool(45) as p:
            labeled_input_output = p.map(labeling, list(distributed_input))
        
        flatten_labeled_input_output = []
        for i in range(len(labeled_input_output)):
            flatten_labeled_input_output += labeled_input_output[i]
        LIO["input"] = list(LIO["input"])
        LIO["output"] = list(LIO["output"])
        LIO["input"] += [pair[0] for pair in flatten_labeled_input_output]
        LIO["output"] += [pair[1] for pair in flatten_labeled_input_output]
        
        print("step 4 end")        
        
        
        ## step 5 - state balancing and slicing
        print("step 5 -data balancing")
        LIO = data_balancing(LIO)
        if len(list(LIO["input"])) > 600000:
            LIO["input"] = LIO["input"][:600000]
            LIO["output"] = LIO["output"][:600000]
        
        ## step 6 - training
        print("step 6 - start training")
        data_info_print(LIO)
        normalized_input = mlp.input_normalize(LIO["input"])
        normalized_input, LIO["input"], LIO["output"] = shuffle(normalized_input, LIO["input"], LIO["output"])
        if len(np.shape(LIO["output"])) < 2:
            LIO["output"] = np.expand_dims(LIO["output"],axis=1)
        mlp.fit(normalized_input,LIO["output"])
        # save
        mlp.model.save_weights("./model_save/model" + str(itr) +".h5")
        print("Saved model to disk")
    
        ## step 7 - NN evaluation from simulation
        print("step 7 - test run start")
        mlp.load_weights_to_dyn(dyn)
        # test_run(dyn, mjc, vis = True, contact = True)
        