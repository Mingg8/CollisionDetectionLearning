import os
import numpy as np
import numpy.linalg as LA
from math import sqrt, floor

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from module.normalize import Normalize
from module.training import Train
from module.file_io import FileIO
from datetime import datetime
from config import config


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
    EPS = 0.00001
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
    i_data, o_data = FileIO.dataLoader(
        file_path + file_name[0],
        'input',
        'output') # 1000k

    i_data = i_data[0:data_num, :]
    o_data = o_data[0:data_num, :]

    sorted_input = np.array(i_data)
    sorted_output = np.array(o_data)

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

    print("i: {}, o: {}".format(np.shape(sorted_input), np.shape(sorted_output)))

    # 3. train
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

    # itr = 0
    # itr_num = 0
    # iteration with different interval of data
    # while (bound < 0.004) :
    #     itr_num += 1
    #     if (itr_num > 10):
    #         break

    #     print("RESULT: \n  b: {}, bound: {}".format(b, bound))
    #     t = Train(
    #         model_save_directory,
    #         figure_save_directory,
    #         error_bound,
    #         model_file,
    #         weight_file
    #         )

    #     if (b > bound - EPS):
    #         # next stage!
    #         prev_bound = bound
    #         bound += 0.0005

    #         for ll in range(last_i, len(sorted_o_data)):
    #             sorted_input.append(sorted_i_data[ll])
    #             sorted_output.append(sorted_o_data[ll])
    #             if sorted_o_data[ll] > bound:
    #                 last_i = i
    #                 break
    #         sorted_input = np.array(sorted_input)
    #         sorted_output = np.array(sorted_output)

    #     # repeat
    #     a, b, left_i, right_i = t.train(
    #         sorted_input,
    #         sorted_output
    #     )


    # iteration with different data sets
    for itr in range(1, len(file_name)):
        t = Train(
            model_save_directory,
            figure_save_directory,
            error_bound,
            model_file,
            weight_file
            )

        new_i, new_o = FileIO.dataLoader(
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

        real_input = []
        real_output = []
        for i in range(len(o_data)):
            if sorted_output[i] > 0.002:
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

    data_num = config["data_num"]
    # data_num = 100
    file_name = [
        '/obj/data/data_final_include_normal.mat'
        ]
    error_bound = config["error_bound"]
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
    


