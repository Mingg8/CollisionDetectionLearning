from multiprocessing import Pool
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as sR
import random
import pickle
from config import config
import module.util as util
from module.evaluation import plot_R3_space_pred, plot_R3_space_input, \
    plot_near_CF_path_pred, plot_near_CF_path_fcl
from copy import deepcopy
import DynamicsCpp
# from module.mjc_module import MjcModule
from module.tree import Tree
from config import config
import numpy.linalg as LA
import functools
from multiprocessing.managers import BaseManager
from math import sqrt

import multiprocessing
multiprocessing.set_start_method('spawn', True)

from sklearn.utils import shuffle
import module.MLP as MLP


class ObjLoader():
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.normals = []
    
    def setObj(self, fileName):
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = [float(line[index1:index2]), float(line[index2:index3]), \
                        float(line[index3:-1])]
                    self.vertices.append(vertex)

                elif line[:2] == "vn":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    normal = [float(line[index1:index2]), float(line[index2:index3]), \
                        float(line[index3:-1])]
                    self.normals.append(normal)

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
                    self.faces.append(face)

            self.vertices = np.array(self.vertices).astype(float)
            self.normals = np.array(self.normals).astype(float)
            self.faces = np.array(self.faces).astype(int)
            
            f.close()
        except IOError:
            print(".obj file not found.")

class CollisionChecker():
    def __init__(self, obj1, obj2):
        self.m1 = fcl.BVHModel()
        self.m1.beginModel(len(obj1.vertices), len(obj1.faces))
        self.m1.addSubModel(obj1.vertices, obj1.faces)
        self.m1.endModel()

        self.m2 = fcl.BVHModel()
        self.m2.beginModel(obj2.vertices.shape[0], obj2.faces.shape[0])
        self.m2.addSubModel(obj2.vertices, obj2.faces)
        self.m2.endModel()
            
        
        R_init = np.array([[1.0, 0.0, 0.0],
                    [0.0,  1.0, 0.0],
                    [0.0,  0.0, 1.0]])
        T_init = np.array([0.0, 0.0, 0.0])

        t = fcl.Transform(R_init, T_init)
        self.obj1 = fcl.CollisionObject(self.m1, t)
        self.obj2 = fcl.CollisionObject(self.m2, t)
    
    def collisioncheck(self):
        request = fcl.CollisionRequest(enable_contact=True)
        result = fcl.CollisionResult()
        ret = fcl.collide(self.obj1, self.obj2, request, result)
        
        if result.contacts == []:
            # return int(~(result.is_collision)) + 1.5
            return (-1, -1)
        else :
            pen_depth = result.contacts[0].penetration_depth
            max_pene_depth = config["max_penetration_depth"] # 1.5 mm
            return (result.contacts[0].b1, result.contacts[0].b2)

def rayIntersectsTriangle(ray_origin, ray_vec, tri):
    EPS = 0.0000001
    tri = np.array(tri).astype(float)
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    h = np.cross(ray_vec, e2)
    a = np.dot(e1, h)
    if (a > -EPS and a < EPS):
        return 0
    f = 1.0 / a
    s = ray_origin - tri[0]
    u = f * np.dot(s, h)
    if (u < 0.0 or u > 1.0):
        return 0
    q = np.cross(s, e1)
    v = f * np.dot(ray_vec, q)
    if (v < 0.0 or u+v>1.0):
        return 0
    t = f * np.dot(e2, q)
    if (t > EPS and t < (1-EPS)):
        pnt = ray_origin + ray_vec * t
        return pnt
    else:
        return 0


def dataGeneration(padding, noise_size):
    obj_mesh = ObjLoader()
    obj_mesh.setObj("obj/nut.obj")
    # mesh order: 0.001

    shape = obj_mesh.vertices.shape
    tetra_vertices = [[0, 0, sqrt(6)/4*padding], \
        [-padding/2, -sqrt(3)/6*padding, -sqrt(6)/12 * padding], \
            [padding/2, -sqrt(3)/6 * padding, -sqrt(6)/12 * padding], \
                [0, sqrt(3)/3 * padding, -sqrt(6)/12 * padding]]

    tetra = ObjLoader()
    tetra.faces = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
    tetra.faces = np.array(tetra.faces).astype(int)

    input_ = []
    output = []
    
    # for i in range(shape[0]):
    for i in range(1000):
        tetra.vertices = []
        noise = np.random.rand(3)

        # make input
        tetra_origin = obj_mesh.vertices[i] + noise * noise_size
        input_.append(tetra_origin)

        # make output
        for j in range(4):
            tetra.vertices.append(tetra_vertices[j] + tetra_origin)
        tetra.vertices = np.array(tetra.vertices).astype(float)
        col = CollisionChecker(tetra, obj_mesh)
        res = col.collisioncheck()
        if res[0] == -1:
            # check if point is inside the mesh
            for l in range(len(obj_mesh.faces)):
                a = [obj_mesh.vertices[ll] for ll in obj_mesh.faces[l]]
                isInInside = rayIntersectsTriangle(np.array(tetra_origin),  \
                    np.array([0, 0, 1]), a)
                if (isInInside is not 0):
                    # point is in inside!!
                    penet = -np.linalg.norm(isInInside - tetra_origin)
                    break
            # point is not colliding
            penet = 1
        else:
            # tetra contact point
            tetra_contact_pnt = np.array([0, 0, 0]).astype(float)
            for l in tetra.faces[res[0]]:
                tetra_contact_pnt += tetra.vertices[l]
            tetra_contact_pnt /= 3

            # obj contact point
            obj_contact_pnt = np.array([0, 0, 0]).astype(float)
            for l in obj_mesh.faces[res[1]]:
                obj_contact_pnt += obj_mesh.vertices[l]
            obj_contact_pnt /= 3

            # normal
            obj_mesh_normal = np.array([0, 0, 0]).astype(float)
            for l in obj_mesh.faces[res[1]]:
                obj_mesh_normal += obj_mesh.normals[l]
            obj_mesh_normal /= np.linalg.norm(obj_mesh_normal)

            penet = np.dot(obj_mesh_normal, \
                tetra_contact_pnt - obj_contact_pnt)
        # print(penet)
        output.append(penet)
        if (i%10 == 0):
            print(i)
    print(output)

def data_info_print(LIO):
    output = np.array(LIO['output'])
    input_ = np.array(LIO['input'])
    idx = np.where(output==0.5)
    out_idx = np.where(output<0)
    num_total_data = len(LIO['output'])
    num_inside_data = np.shape(output[idx])[0]
    num_outsize_data1 = num_total_data - num_inside_data
    print("cur data - total_data {} // inside data : {} // \
        outside data : {}".format(num_total_data, num_inside_data, \
            num_total_data - num_inside_data))
    
    new_input = np.array(LIO['input'])[idx]
    
    deep_nut_idx = np.where(new_input[:,2] < 0.0445)
    num_total_data = len(new_input)
    num_inside_data = np.shape(input_[deep_nut_idx])[0]
    num_outside_data = num_total_data - num_inside_data
    print("valid data - total_data {} // inside data : {} // outside data : \
        {}".format(num_total_data, num_inside_data, num_total_data - \
            num_inside_data))

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
    
    print("before - total_data {} // inside data : {} // outside data : \
        {}".format(num_total_data, num_inside_data, num_outsize_data))
    
    if (num_outsize_data > num_inside_data*1.1):
        input_outside, output_outside = shuffle(input_outside, output_outside)
        input_outside, output_outside = input_outside[:num_inside_data], \
            output_outside[:num_inside_data]
    
    LIO['output'] = list(output_inside) + list(output_outside)
    LIO['input'] =  list(input_inside) + list(input_outside)
    
    print("after - total_data {} // inside data : {} // outside data : {} \
        ".format(len(LIO['output']), len(list(output_inside)), \
            len(list(output_outside))))
    
    LIO['output'], LIO['input'] = shuffle(np.array(LIO['output']), \
        np.array(LIO['input']))
    
    return LIO


def artificial_nominal_data():
    print("plot near collision free path")
    resolution = 15000
    total_angle = 4.2*np.pi
    start_i = int((140/180*np.pi)*resolution/total_angle)
    print(start_i)
    states = []
    for i in range(start_i,resolution):
        states.append(util.collision_free_config_gen(-i/resolution*total_angle, \
            output_type=1))
    
    states += (list(np.array(states) + np.array([0,0,0.0001,0,0,0]))+
        list(np.array(states) + np.array([0,0,-0.0001,0,0,0]))+
        list(np.array(states) + np.array([0.0001,0,0,0,0,0]))+
        list(np.array(states) + np.array([-0.0001,0,0,0,0,0]))
        )
    return states

def test_run(dyn, mjc, vis = True, contact = False):
    state = np.array(list(util.initial_pos_distribution()) + [0,0,0,0,0,0])
    pos_stack_tmp = []
    for i in range(2500):
        u = tree.tree_follower(state)
        u += 0.001*40/(config["substep"] * config["dt"])  * np.random.normal( \
            0,1,[6]) * np.array([1,1,1,0.5,0.5,0.5])
        for i in range(config["substep"]):
            if contact:
                state = dyn.fwd_dynamics(state, u)
            else:
                state = dyn.fwd_dynamics_wo_contact(state, u)
        pos_stack_tmp.append(state[:6])
        if LA.norm(state[:6] - config["goal"]) < 0.030:
            state[:6] = util.initial_pos_distribution()
            break
        if vis:
            mjc.vis_cur_pos(state[:6])
    return pos_stack_tmp[int(len(pos_stack_tmp)/2):]

    
def concentration(input, mlp):
    label_pred = mlp.predict(np.array(input))
    
    prob_dist = 2*np.absolute(label_pred) - 0.01
    
    rej_criteria = np.random.rand(len(list(input)))
    important_pnts_idx = np.where(prob_dist<=rej_criteria)
    
    return np.array(input)[important_pnts_idx]

def labeling(inputs):
    CD = CollisionChecker()
    return [[input, CD.collisioncheck_from_state(np.array(input))] for \
        input in inputs]

def checkCollisionDetection():
    padding = 1
    tetra_vertices = [[0, 0, sqrt(6)/4*padding], \
        [-padding/2, -sqrt(3)/6*padding, -sqrt(6)/12 * padding], \
            [padding/2, -sqrt(3)/6 * padding, -sqrt(6)/12 * padding], \
                [0, sqrt(3)/3 * padding, -sqrt(6)/12 * padding]]

    tetra = ObjLoader()
    tetra.faces = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
    tetra.faces = np.array(tetra.faces).astype(int)
    tetra.vertices = np.array(tetra_vertices).astype(float)

    offset = np.array([0.0, 0.1, 0.1])

    padding = 0.1
    tetra_vertices = [[0, 0, sqrt(6)/4*padding], \
        [-padding/2, -sqrt(3)/6*padding, -sqrt(6)/12 * padding], \
            [padding/2, -sqrt(3)/6 * padding, -sqrt(6)/12 * padding], \
                [0, sqrt(3)/3 * padding, -sqrt(6)/12 * padding]]

    tetra_small = ObjLoader()
    tetra_small.faces = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
    tetra_small.faces = np.array(tetra.faces).astype(int)
    tetra_small.vertices = np.array(tetra_vertices).astype(float)
    for i in range(len(tetra_small.vertices)):
        tetra_small.vertices[i] += offset

    col = CollisionChecker(tetra, tetra_small)
    res = col.collisioncheck()
    print(res)

    for i in range(len(tetra.faces)):
        a = [tetra.vertices[j] for j in tetra.faces[i]]
        isInInside = rayIntersectsTriangle(offset, np.array([0, 0, 1]),\
            a)
        print(isInInside)

if __name__ == "__main__":
    dataGeneration(0.01, 0.01) # padding, noise_size
    # 0.01, 0.001
    dataGeneration(0.001, 0.001)

    mlp = MLP.MLP(config, hdim=64)
    tree = Tree("graph_data/graph_GraphML.xml", config)
    
    # learning continue! for load weight
    mlp.load_model('model_save/model0.h5')
    
    bound = config["bound"]
    LIO = {}
    LIO["state"] = []    
    LIO["input"] = []
    LIO["output"] = []
    
    for itr in range(100):
        print("===== iteration {} start".format(itr))
        ## step 1 - exploration and data generation from demonstration
        print("step 1 - data generation by tree-follower")
        # data from prior iteration
        LIO["input"] = list(LIO["input"])
        LIO["output"] = list(LIO["output"])

        # TODO: change new_input
        new_input = []
        with Pool(45) as p:
            # generated_pos_data = p.map(data_gen_from_simulation, range(200))
            generated_pos_Data = p.map(data_gen, range(200))
            
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
        normalized_input, LIO["input"], LIO["output"] = shuffle(normalized_input, \
            LIO["input"], LIO["output"])
        if len(np.shape(LIO["output"])) < 2:
            LIO["output"] = np.expand_dims(LIO["output"],axis=1)
        mlp.fit(normalized_input,LIO["output"])
        # save
        mlp.model.save_weights("./model_save/model" + str(itr) +".h5")
        print("Saved model to disk")
    
        ## step 7 - NN evaluation from simulation
        print("step 7 - test run start")
        # mlp.load_weights_to_dyn(dyn) # weight update
        # test_run(dyn, mjc, vis = True, contact = True)
        