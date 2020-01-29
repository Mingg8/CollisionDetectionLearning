import os
from math import sqrt, floor
from datetime import datetime
import numpy as np
import numpy.linalg as LA

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from module.normalize import Normalize
from module.training import Train
from module.file_io import FileIO
from module.data_processing import DataProcessing
from config import config


def training(
    error_bound,
    file_path,
    save_directory,
    file_name,
    data_num,
    new = True,
    m_file_name = "",
    w_file_name = ""
    ):
    EPS = 0.00001
    if new:
        t = Train(
            save_directory,
            error_bound
            )
    else:
        t = Train(
            save_directory,
            error_bound,
            m_file_name = m_file_name,
            w_file_name= w_file_name
        )
    i_data, o_data = FileIO.dataLoader(
        file_path + file_name[0],
        'input',
        'output')
    data = DataProcessing(i_data, o_data, data_num, save_directory)

    ##################### DATA PREPROCESSING ###################### 
    # 1. sorting data
    # sorted_ind = sorted(range(len(o_data)), key = lambda k:o_data[k][0])
    # sorted_i_data = np.array([i_data[i] for i in sorted_ind])
    # sorted_o_data = np.array([o_data[i] for i in sorted_ind])

    # # 2. cut data
    # prev_bound = -0.002
    # bound = 0.000
    # sorted_input = []
    # sorted_output = []

    # for i in range(len(sorted_o_data)):
    #     if sorted_o_data[i] > bound:
    #         last_i = i
    #         break
    #     sorted_input.append(sorted_i_data[i])
    #     sorted_output.append(sorted_o_data[i])

    # for i in range(len(sorted_o_data)-1, -1, -1):
    #     if sorted_o_data[i, 0] < 0:
    #         break;
    #     sorted_input.append(sorted_i_data[i])
    #     sorted_output.append(sorted_o_data[i])

    # sorted_input = np.array(sorted_input)
    # sorted_output = np.array(sorted_output)
    ##################### DATA PREPROCESSING ###################### 

    print("i: {}, o: {}".format(np.shape(data.train_input), \
        np.shape(data.train_output)))

    # 3. train
    a, b, left_i, right_i = t.train(
        data
        )
    model_file = "/model0.json"
    weight_file = "/weight0.h5"
    t.saveFile(
        model_file,
        weight_file
        )
    del t

    # iteration with different data sets
    for itr in range(1, len(file_name)):
        t = Train(
            save_directory,
            error_bound,
            m_file_name = model_file,
            w_file_name = weight_file
            )

        new_i, new_o = FileIO.dataLoader(
            file_path + file_name[itr],
            'input',
            'output') # 1000k
            
        a, b, left_i, right_i = t.train(
            data
            )
        
        model_file = "/model"+str(itr) +".json"
        weight_file = "/weight"+str(itr)+".h5"
        t.saveFile(
            model_file,
            weight_file
            )
        del t

if __name__ == "__main__":
    file_path = str(os.getcwd())
    now = datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H:%M")
    # now_string = "2020-01-17_14:33"
    save_directory = file_path + '/old_results/' + now_string
    try:
        os.makedirs(save_directory)
    except:
        print("already exists")

    ############ LOAD & TEST
    # m_file = "/old_results/0117:12:00/model6.json"
    # w_file = "/old_results/0117:12:00/model6.h5"
    # t = Train(
    #     m_file_name = file_path + m_file,
    #     w_file_name = file_path + w_file,
    #     save_directory = save_directory
    #     )
    # t.loadAndTest(file_path)

    ############ TRAINING
    try:
        os.makedirs(save_directory)
    except:
        print("already exists")

    data_num = config["data_num"]
    file_name = config["file_name"]
    error_bound = config["error_bound"]
    training(
        error_bound,
        file_path,
        save_directory,
        file_name,
        data_num
        )
    
    ############ LOAD & TRAIN
    # now_string = "2020-01-17_14:33"
    # save_directory = file_path + '/old_results/' + now_string
    # model_file_name = "/model0.json"
    # weight_file_name = "/model0.h5"

    # try:
    #     os.makedirs(save_directory)
    # except:
    #     print("already exists")

    # data_num = 1000000
    # file_name = [
    #     '/obj/data/data_final1.mat',
    #     '/obj/data/data_final1.mat',
    #     '/obj/data/data_final1.mat'
    #     ]
    # error_bound = 0.002
    # training(
    #     error_bound,
    #     file_path,
    #     save_directory,
    #     file_name,
    #     data_num,
    #     False,
    #     model_file_name,
    #     weight_file_name
    #     )
    


