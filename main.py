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

def loadModelAndTest(m_file, w_file):
    t = Train(
        m_file_name = file_path + m_file,
        w_file_name = file_path + w_file,
        save_directory = save_directory
        )
    t.loadAndTest(file_path)


def loadModelAndSaveCsv(save_dir, model_file_name, weight_file_name):
    t = Train(
        save_dir,
        0.03,
        model_file_name,
        weight_file_name
    )
    t.saveWeightAsCsv()

def training(
    error_bound,
    file_path,
    save_directory,
    file_name,
    data_num,
    m_file_name = "",
    w_file_name = ""
    ):
    EPS = 0.00001
    if m_file_name == "" and w_file_name == "":
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

    print("i: {}, o: {}".format(np.shape(data.train_input), \
        np.shape(data.train_output)))

    # 3. train
    t.train(data)

    now = datetime.now()
    now_string = now.strftime("_%Y-%m-%d_%H:%M")

    model_file = "/model" + now_string + ".json"
    weight_file = "/weight" + now_string +".h5"
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
            
        t.train(data)
        
        model_file = "/model"+str(itr) +".json"
        weight_file = "/weight"+str(itr)+".h5"
        t.saveFile(
            model_file,
            weight_file
            )
        del t

if __name__ == "__main__":
    # file_path = str(os.getcwd())
    # now = datetime.now()
    # now_string = now.strftime("%Y-%m-%d_%H:%M")
    # save_directory = file_path + '/old_results/' + now_string
    # try:
    #     os.makedirs(save_directory)
    # except:
    #     print("already exists")

    # ############ TRAINING
    # data_num = config["data_num"]
    # file_name = config["file_name"]
    # error_bound = config["error_bound"]
    # training(
    #     error_bound,
    #     file_path,
    #     save_directory,
    #     file_name,
    #     data_num
    #     )

    save_dir = '/home/mjlee/workspace/NutLearning/old_results/2020-01-29_20:02_hdim_64'
    model_file_name = '/model3.json'
    weight_file_name = '/weight3.h5'
    loadModelAndSaveCsv(save_dir, model_file_name, weight_file_name)

    # training(error_bound,
    #   file_path,
    #   file_path + '/old_results/2020-01-29_20:02_hdim_64',
    #   file_name,
    #   data_num,
    #   '/model3.json',
    #   '/weight3.h5')
