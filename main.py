# from multiprocessing import Pool
import numpy as np
# import fcl
# from scipy.spatial.transform import Rotation as sR
from scipy.io import loadmat
# import random
# import pickle
# from config import config
# import module.util as util
# from copy import deepcopy
# import DynamicsCpp
# from module.mjc_module import MjcModule
# from module.tree import Tree
import numpy.linalg as LA
# import functools
# from multiprocessing.managers import BaseManager
from math import sqrt, floor

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# class ObjLoader():
#     def __init__(self):
#         self.vertices = []
#         self.faces = []
#         self.normals = []
#         self.face_normals = []
    
#     def calFaceNormals(self):
#         for i in range(len(self.faces)):
#             face_pnts += [self.vertices[l] for l in self.faces[i]]

    
#     def setObj(self, fileName):
#         try:
#             f = open(fileName)
#             for line in f:
#                 if line[:2] == "v ":
#                     index1 = line.find(" ") + 1
#                     index2 = line.find(" ", index1 + 1)
#                     index3 = line.find(" ", index2 + 1)

#                     vertex = [float(line[index1:index2]), float(line[index2:index3]), \
#                         float(line[index3:-1])]
#                     self.vertices.append(vertex)

#                 elif line[:2] == "vn":
#                     index1 = line.find(" ") + 1
#                     index2 = line.find(" ", index1 + 1)
#                     index3 = line.find(" ", index2 + 1)

#                     normal = [float(line[index1:index2]), float(line[index2:index3]), \
#                         float(line[index3:-1])]
#                     self.normals.append(normal)

#                 elif line[0] == "f":
#                     string = line.replace("//", "/")
#                     ##
#                     i = string.find(" ") + 1
#                     face = []
#                     for item in range(string.count(" ")):
#                         if string.find(" ", i) == -1:
#                             face.append(int(string[i:-1].split("/")[0])-1)
#                             break
#                         face.append(int(string[i:string.find(" ", i)].split("/")[0])-1)
#                         i = string.find(" ", i) + 1
#                     ##
#                     self.faces.append(face)

#             self.vertices = np.array(self.vertices).astype(float)
#             self.normals = np.array(self.normals).astype(float)
#             self.faces = np.array(self.faces).astype(int)
            
#             f.close()
#         except IOError:
#             print(".obj file not found.")

def dataLoader(filename):
    mat_contents = loadmat(filename)
    i_data = mat_contents['input']
    o_data = mat_contents['output']
    return i_data, o_data

# def data_info_print(LIO):
#     output = np.array(LIO['output'])
#     input_ = np.array(LIO['input'])
#     idx = np.where(output==0.5)
#     out_idx = np.where(output<0)
#     num_total_data = len(LIO['output'])
#     num_inside_data = np.shape(output[idx])[0]
#     num_outsize_data1 = num_total_data - num_inside_data
#     print("cur data - total_data {} // inside data : {} // \
#         outside data : {}".format(num_total_data, num_inside_data, \
#             num_total_data - num_inside_data))
    
#     new_input = np.array(LIO['input'])[idx]
    
#     deep_nut_idx = np.where(new_input[:,2] < 0.0445)
#     num_total_data = len(new_input)
#     num_inside_data = np.shape(input_[deep_nut_idx])[0]
#     num_outside_data = num_total_data - num_inside_data
#     print("valid data - total_data {} // inside data : {} // outside data : \
#         {}".format(num_total_data, num_inside_data, num_total_data - \
#             num_inside_data))

# def data_balancing(LIO):
#     output = np.array(LIO['output'])
#     input_ = np.array(LIO['input'])
#     in_idx = np.where(output==0.5)
#     out_idx = np.where(output<0)
    
#     input_inside = input_[in_idx]
#     output_inside = output[in_idx]
#     input_outside = input_[out_idx]
#     output_outside = output[out_idx]
    
#     num_total_data = len(LIO['output'])
#     num_inside_data = np.shape(output_inside)[0]
#     num_outsize_data = np.shape(output_outside)[0]
    
#     print("before - total_data {} // inside data : {} // outside data : \
#         {}".format(num_total_data, num_inside_data, num_outsize_data))
    
#     if (num_outsize_data > num_inside_data*1.1):
#         input_outside, output_outside = shuffle(input_outside, output_outside)
#         input_outside, output_outside = input_outside[:num_inside_data], \
#             output_outside[:num_inside_data]
    
#     LIO['output'] = list(output_inside) + list(output_outside)
#     LIO['input'] =  list(input_inside) + list(input_outside)
    
#     print("after - total_data {} // inside data : {} // outside data : {} \
#         ".format(len(LIO['output']), len(list(output_inside)), \
#             len(list(output_outside))))
    
#     LIO['output'], LIO['input'] = shuffle(np.array(LIO['output']), \
#         np.array(LIO['input']))
    
#     return LIO


# def artificial_nominal_data():
#     print("plot near collision free path")
#     resolution = 15000
#     total_angle = 4.2*np.pi
#     start_i = int((140/180*np.pi)*resolution/total_angle)
#     print(start_i)
#     states = []
#     for i in range(start_i,resolution):
#         states.append(util.collision_free_config_gen(-i/resolution*total_angle, \
#             output_type=1))
    
#     states += (list(np.array(states) + np.array([0,0,0.0001,0,0,0]))+
#         list(np.array(states) + np.array([0,0,-0.0001,0,0,0]))+
#         list(np.array(states) + np.array([0.0001,0,0,0,0,0]))+
#         list(np.array(states) + np.array([-0.0001,0,0,0,0,0]))
#         )
#     return states

# def test_run(dyn, mjc, vis = True, contact = False):
#     state = np.array(list(util.initial_pos_distribution()) + [0,0,0,0,0,0])
#     pos_stack_tmp = []
#     for i in range(2500):
#         u = tree.tree_follower(state)
#         u += 0.001*40/(config["substep"] * config["dt"])  * np.random.normal( \
#             0,1,[6]) * np.array([1,1,1,0.5,0.5,0.5])
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

    
# def concentration(input, mlp):
#     label_pred = mlp.predict(np.array(input))
    
#     prob_dist = 2*np.absolute(label_pred) - 0.01
    
#     rej_criteria = np.random.rand(len(list(input)))
#     important_pnts_idx = np.where(prob_dist<=rej_criteria)
    
#     return np.array(input)[important_pnts_idx]

# def labeling(inputs):
#     CD = CollisionChecker()
#     return [[input, CD.collisioncheck_from_state(np.array(input))] for \
#         input in inputs]

# def checkCollisionDetection():
#     padding = 1
#     tetra_vertices = [[0, 0, sqrt(6)/4*padding], \
#         [-padding/2, -sqrt(3)/6*padding, -sqrt(6)/12 * padding], \
#             [padding/2, -sqrt(3)/6 * padding, -sqrt(6)/12 * padding], \
#                 [0, sqrt(3)/3 * padding, -sqrt(6)/12 * padding]]

#     tetra = ObjLoader()
#     tetra.faces = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
#     tetra.faces = np.array(tetra.faces).astype(int)
#     tetra.vertices = np.array(tetra_vertices).astype(float)

#     offset = np.array([0.0, 0.1, 0.1])

#     padding = 0.1
#     tetra_vertices = [[0, 0, sqrt(6)/4*padding], \
#         [-padding/2, -sqrt(3)/6*padding, -sqrt(6)/12 * padding], \
#             [padding/2, -sqrt(3)/6 * padding, -sqrt(6)/12 * padding], \
#                 [0, sqrt(3)/3 * padding, -sqrt(6)/12 * padding]]

#     tetra_small = ObjLoader()
#     tetra_small.faces = [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]
#     tetra_small.faces = np.array(tetra.faces).astype(int)
#     tetra_small.vertices = np.array(tetra_vertices).astype(float)
#     for i in range(len(tetra_small.vertices)):
#         tetra_small.vertices[i] += offset

#     col = CollisionChecker(tetra, tetra_small)
#     res = col.collisioncheck()
#     print(res)

#     for i in range(len(tetra.faces)):
#         a = [tetra.vertices[j] for j in tetra.faces[i]]
#         in_inside = rayIntersectsTriangle(offset, np.array([0, 0, 1]),\
#             a)
#         print(in_inside)

if __name__ == "__main__":
    file_path = '/home/mjlee/workspace/NutLearning/obj/'
    filename = 'final_data.mat'
    i_data, o_data = dataLoader(file_path + filename)

    print(np.shape(i_data))
    print(np.shape(o_data))

    # training
    model = Sequential()
    model.add(Dense(5, input_dim = 3, activation = 'relu'))
    model.add(Dense(3, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', \
        metric = ['accuracy'])
    model.fit(i_data, o_data, epochs = 150, batch_size = 10)
    _, accuracy = model.evaluate(i_data, o_data)
    print('Accuracy: %.2f' % (accuracy * 100))

    # # predict
    # predictions = model.predict(i_data)
    # # round predictions
    # rounded = [round(x[0]) for x in predictions]
