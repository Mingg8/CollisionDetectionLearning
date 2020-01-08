# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import os, sys
sys.path.insert(0, os.path.abspath("/home/dongwon/research/configuration_learning_nut"))
import numpy as np
# import src.learning.MLP as MLP
# import src.learning.MLP_add_state as MLP
import pickle
from sklearn.utils import shuffle
import math
from multiprocessing import Process


from evaluation import plot_after_conversion_add_goal
from evaluation import plot_before_conversion
from util import construct_env

def data_load():
    with open('./data_save/path_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
 
    MLP_input = []
    MLP_output = []
    for i in range(len(raw_data)):
        MLP_input = MLP_input + [list(dp['point1']) + list(dp['point2'])+list(dp['root'])+list(dp['end']) for dp in raw_data[i][2]]
        MLP_output =MLP_output + [[dp['pathlen']] for dp in raw_data[i][2]]
    
    return MLP_input, MLP_output, raw_data


def curriculum_training(mlp, progress_idx, MLP_input, MLP_output):
    max_size = 300000
    MLP_input, MLP_output = shuffle(MLP_input, MLP_output)
    w = [0.05, 0.2, 1*math.exp(-progress_idx), 10]
    print("curriculum learning {} th // loss_w : {}".format(progress_idx, w))
    weights = mlp.model.get_weights()
    mlp.compile_model(w)
    mlp.model.set_weights(weights)
    mlp.fit(MLP_input[:max_size,:], MLP_output[:max_size,:])
    # save
    mlp.model.save_weights("./model_save/model_add_input.h5")
    print("Saved model to disk")


if __name__=="__main__":
      
    config = dict()
    config['system_dim'] = 2
    config['w_init'] = [0.005, 0.05, 0.5, 1] # w[0]*parallel_loss(y_true, y_pred) + w[1]*pathlen_loss(y_true, y_pred) + w[2]*fixpnts_loss(y_true, y_pred) + w[3]*line_loss(y_true, y_pred)
    mlp = MLP.MLP(config)
    
    MLP_input, MLP_output, raw_data = data_load()
 
    print("input dim : {}".format(len(MLP_input)))

    MLP_input = shuffle(mlp.input_normalize(np.array(MLP_input)))
    MLP_output = shuffle(mlp.output_normalize(np.array(MLP_output)))
    
    print("input shape : {} // output shape : {}".format(np.shape(MLP_input), np.shape(MLP_output)))
 
    for i in range(4):
        MLP_input, MLP_output = shuffle(MLP_input, MLP_output)
    
        curriculum_training(mlp, i, MLP_input, MLP_output)
        plot_after_conversion_add_goal(mlp, raw_data)