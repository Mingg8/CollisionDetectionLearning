import os
import numpy as np
import numpy.linalg as LA
from math import sqrt, floor

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from module.normalize import Normalize
from module.training import Train
from datetime import datetime


def training(
    error_bound,
    file_path,
    figure_save_directory,
    model_save_directory,
    file_name,
    data_num,
    new = True,
    m_file_name = "",
    w_file_name = ""
    ):
    if new:
        t = Train(
            model_save_directory,
            figure_save_directory,
            error_bound
            )
    else:
        t = Train(
            model_save_directory,
            figure_save_directory,
            error_bound,
            m_file_name = m_file_name,
            w_file_name= w_file_name
        )
    i_data, o_data = t.dataLoader(
        file_path + file_name[0],
        'input',
        'output') # 1000k

    i_data = i_data[1:data_num, :]
    o_data = o_data[1:data_num, :]

    sorted_ind = sorted(range(len(o_data)), key = lambda k:o_data[k])
    sorted_input = np.array([i_data[i] for i in sorted_ind])
    sorted_output = np.array(sorted(o_data))

    real_input = []
    real_output = []
    for i in range(len(o_data)):
        if sorted_output[i] > 0.004:
            break
        if sorted_output[i] > -0.002:
            real_input.append(sorted_input[i])
            real_output.append(sorted_output[i])
    sorted_input = np.array(real_input)
    sorted_output = np.array(real_output)
    del real_output, real_input

    a, b, left_i, right_i = t.train(
        sorted_input,
        sorted_output
        )

    now = datetime.now()
    now_string = now.strftime("_%H:%M")
    model_file = "/model" + now_string + ".json"
    weight_file = "/model" + now_string + ".h5"
    t.saveFile(
        model_file,
        weight_file
        )
    del t

    for itr in range(1, len(file_name)):
        t = Train(
            model_save_directory,
            figure_save_directory,
            error_bound,
            model_file,
            weight_file
            )
        new_i, new_o = t.dataLoader(
            file_path + file_name[itr],
            'input',
            'output') # 1000k

        change_num = 0
        for i in range(len(new_o)):
            if (new_o[i] < a or new_o[i] > b):
                if change_num < right_i - left_i :    
                    index = left_i + int((right_i - left_i) * np.random.rand(1))
                else:
                    index = int(np.random.rand(1) * len(new_o))
                sorted_input[index] = new_i[i]
                sorted_output[index] = new_o[i]
                change_num += 1

        print("changing: {}".format(change_num))
        del new_i, new_o

        sorted_ind = sorted(range(len(sorted_output)), key = lambda k:sorted_output[k])
        sorted_input = np.array([sorted_input[i] for i in sorted_ind])
        sorted_output = np.array(sorted(sorted_output))

        # real_input = []
        # real_output = []
        # for i in range(len(o_data)):
        #     if sorted_output[i] > 0.002:
        #         break
        #     if sorted_output[i] > -0.002:
        #         real_input.append(sorted_input[i])
        #         real_output.append(sorted_output[i])
        # sorted_input = np.array(real_input)
        # sorted_output = np.array(real_output)
        # del real_output, real_input
            
        a, b, left_i, right_i = t.train(
            sorted_input,
            sorted_output
            )
        
        now = datetime.now()
        now_string = now.strftime("_%H:%M")
        model_file = "/model" + now_string + ".json"
        weight_file = "/model" + now_string + ".h5"
        t.saveFile(
            model_save_directory + model_file,
            model_save_directory + weight_file
            )
        del t

if __name__ == "__main__":
    file_path = str(os.getcwd())
    now = datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H:%M")
    # now_string = "2020-01-17_14:33"
    model_save_directory = file_path + '/old_results/' + now_string
    figure_save_directory = file_path + '/figure/' + now_string
    try:
        os.makedirs(figure_save_directory)
    except:
        print("already exists")

    ############ LOAD & TEST
    # m_file = "/old_results/0117:12:00/model6.json"
    # w_file = "/old_results/0117:12:00/model6.h5"
    # t = Train(
    #     m_file_name = file_path + m_file,
    #     w_file_name = file_path + w_file,
    #     save_directory = figure_save_directory
    #     )
    # t.loadAndTest(file_path)

    ############ TRAINING
    try:
        os.makedirs(model_save_directory)
    except:
        print("already exists")

    data_num = 1107991
    file_name = [
        '/obj/data/data_broad.mat',
        # '/obj/data/data_final1.mat',
        # '/obj/data/data_final1.mat'
        # '/obj/data/data_final2.mat',
        # '/obj/data/data_final3.mat',
        # '/obj/data/data_final4.mat',
        # '/obj/data/data_final5.mat'
        ]
    error_bound = 0.001
    training(
        error_bound,
        file_path,
        figure_save_directory,
        model_save_directory,
        file_name,
        data_num
        )
    
    ############ LOAD & TRAIN
    # now_string = "2020-01-17_14:33"
    # model_save_directory = file_path + '/old_results/' + now_string
    # figure_save_directory = file_path + '/figure/' + now_string
    # model_file_name = "/model0.json"
    # weight_file_name = "/model0.h5"

    # try:
    #     os.makedirs(model_save_directory)
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
    #     figure_save_directory,
    #     model_save_directory,
    #     file_name,
    #     data_num,
    #     False,
    #     model_file_name,
    #     weight_file_name
    #     )
    


