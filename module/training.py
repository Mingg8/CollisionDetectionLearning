import sys
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

from module.normalize import Normalize
from datetime import datetime
from config import config
from module.file_io import FileIO

class Train:
    def __init__(self,
        model_save_directory,
        save_directory,
        err = 0.002,
        m_file_name = "",
        w_file_name = ""
        ):
        self.error_bound = err
        self.figure_save_dir = save_directory
        self.model_save_dir = model_save_directory
        print("dir: "+self.model_save_dir)

        tf.compat.v1.disable_eager_execution()
        if (m_file_name == "" and w_file_name == ""):
            self.model = Sequential()
            # training
            self.model.add(Dense(256, input_dim = 3, activation = 'relu'))
            self.model.add(Dense(256, activation = 'relu'))
            self.model.add(Dense(256, activation = 'relu'))
            self.model.add(Dense(4, activation = 'tanh'))

        else:
            self.model = FileIO.loadfile(self.model_save_dir, m_file_name, w_file_name)

        self.predict_func = K.function(
            self.model.input,
            self.model.output)

        self.grad = keras.layers.Lambda(
            lambda z: K.gradients(z[0][0, :], z[1]))\
                ([self.model.output, self.model.input])
        self.grad_calc = K.function(self.model.input, self.grad)
        
    def train(self, i_data, o_data):
        train_i, train_o, valid_i ,valid_o, test_i, test_o = \
            self.dataSplit(i_data, o_data, 0.7, 0.15, 0.15)

        n = Normalize(train_i, train_o)
        train_input, train_output = n.dataNormalize(train_i, train_o)
        valid_input, valid_output = n.dataNormalize(valid_i, valid_o)
        test_input, test_output = n.dataNormalize(test_i, test_o)

        self.compile_model(config["weight"])
        self.model.fit(train_input, train_output,
            shuffle = True,
            epochs = config["epoch"],
            batch_size = config["batch"],
            validation_data = (valid_input, valid_output)
            )

        min, max, left_i, right_i = self.evaluation(
            test_input,
            test_output,
            n)
        return min, max, left_i ,right_i
    
    def compile_model(self, w):
        def mse_loss(y_true, y_pred):
            mse_loss = w[0] * K.mean(K.square(y_true[:,0]-y_pred[:,0]))
            return mse_loss
        
        def grad_loss(y_true, y_pred):
            grad_normal_loss = w[1] * K.mean(tf.math.acos(K.sum(
                tf.math.l2_normalize(self.grad) * y_true[:,1:4], axis = 1)))
            return grad_normal_loss
        
        def tot_loss(y_true, y_pred):
            return mse_loss(y_true, y_pred) + grad_loss(y_true, y_pred)

        self.model.compile(
            loss = tot_loss,
            optimizer = 'adam',
            metrics = [mse_loss, grad_loss]
            # metrics = [mse_loss]
        )

    
    def saveFile(self, m_file_name, w_file_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.model_save_dir + m_file_name, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(self.model_save_dir + w_file_name)
        print("Save model to disk, name: " + self.model_save_dir
            + m_file_name)

    def evaluation(self, i_data, o_data, n):
        # predict
        # prediction = self.model.predict(i_data)
        prediction = np.squeeze(self.predict_func(i_data)) # normalized
        grad_pred = np.squeeze(self.grad_calc(i_data)) # normalized

        predict_output = n.oDataUnnormalize(prediction)
        predict_output = predict_output[:, 0]
        real_output = n.oDataUnnormalize(o_data)
        real_output = real_output[:, 0]
        real_input = n.iDataUnnormalize(i_data)
        predict_grad = n.gDataUnnormalize(grad_pred)

        FileIO.saveData(self.model_save_dir, "/data.csv", real_input, real_output,
            predict_output, predict_grad, grad_pred)

        sorted_ind = sorted(range(len(real_output)), key = lambda k:real_output[k])
        sorted_prediction = np.array([predict_output[i] for i in sorted_ind])
        sorted_output = np.array([real_output[i] for i in sorted_ind])

        error = abs(sorted_prediction - sorted_output)
        
        print("max error: %.10f" %(np.max(error)))

        x = range(len(sorted_output))
        plt.plot(x, sorted_prediction, 'b', sorted_output, 'r')
        now = datetime.now()
        now_string = now.strftime("%_Y-%m-%d_%H:%M")
        plt.savefig(self.figure_save_dir +'/figure'+now_string+ '.png')
        plt.clf()
        # plt.show()

        for idx in range(len(sorted_output)):
            if sorted_output[idx] > -0.004:
                break
        left_i = 0
        for i in range(idx, -1, -1):
            if error[i] > self.error_bound:
                left_i = i
                break
        print("left i: %d" %left_i)
        right_i = len(sorted_output) - 1
        for j in range(idx, len(sorted_output)):
            if error[j] > self.error_bound:
                right_i = j
                break
        print("right i: %d" %right_i)
        print("bound: {} ~ {}".format(sorted_output[left_i], sorted_output[right_i]))

        right = np.zeros(100).astype(float)
        wrong= np.zeros(100).astype(float)
        for i in range(idx, len(sorted_output)):
            if sorted_output[i] > 0.005:
                break
            if sorted_output[i] >= 0:
                section = int(sorted_output[i] * 20000)
                if sorted_prediction[i] > 0:
                    right[section] += 1
                else:
                    wrong[section] += 1
        accuracy = np.zeros(100)
        for i in range(100):
            if (wrong[i] + right[i] > 0):
                accuracy[i] = (right[i] / (wrong[i] + right[i]))
            else:
                accuracy[i] = 0
        X = np.arange(0, 0.005, 0.005 / 100)
        plt.plot(X, accuracy)
        # plt.xticks(np.arange(0, 0.005, 0.0002))
        plt.rc('axes', labelsize = 4)
        plt.grid()
        plt.savefig(self.figure_save_dir + '/figure2.png')
        plt.clf()
        return sorted_output[left_i], sorted_output[right_i], left_i, right_i

    def loadAndTest(self, file_path):
        filename = '/obj/data/data_final1.mat'
        i_data, o_data = FileIO.dataLoader(
            file_path + filename,
            'input',
            'output'
            )
        _, _, _, _, test_i, test_o = self.dataSplit(i_data, o_data)
        n = Normalize(test_i, test_o)
        test_input, test_output = n.dataNormalize(test_i, test_o)
        
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam',
            metric = ['accuracy'])
        self.evaluation(test_input, test_output, n)


    def dataSplit(self, i_data, o_data, tr = 0.7, va = 0.15, te = 0.15):
        train_i = []
        train_o = []
        test_i = []
        test_o = []
        valid_i = []
        valid_o = []

        for i in range(len(o_data)):
            if i % 100 < int(tr * 100) :
                train_i.append(i_data[i, :])
                train_o.append(o_data[i, :])
            elif i % 100 < int((tr + va) * 100):
                valid_i.append(i_data[i, :])
                valid_o.append(o_data[i, :])
            else :
                test_i.append(i_data[i, :])
                test_o.append(o_data[i, :])

        train_i = np.array(train_i)
        train_o = np.array(train_o)
        test_i = np.array(test_i)
        test_o = np.array(test_o)
        valid_i = np.array(valid_i)
        valid_o = np.array(valid_o)
        return train_i, train_o, valid_i, valid_o, test_i, test_o

    def FindBatchSize(model):
        """#model: model architecture, that is yet to be trained"""
        import os, sys, psutil, gc, tensorflow, keras
        import numpy as np
        from keras import backend as K
        BatchFound= 16

        try:
            total_params= int(model.count_params());    GCPU= "CPU"
            #find whether gpu is available
            try:
                if K.tensorflow_backend._get_available_gpus()== []:
                    GCPU= "CPU";    #CPU and Cuda9GPU
                else:
                    GCPU= "GPU"
            except:
                from tensorflow.python.client import device_lib;    #Cuda8GPU
                def get_available_gpus():
                    local_device_protos= device_lib.list_local_devices()
                    return [x.name for x in local_device_protos if x.device_type == 'GPU']
                if "gpu" not in str(get_available_gpus()).lower():
                    GCPU= "CPU"
                else:
                    GCPU= "GPU"

            #decide batch size on the basis of GPU availability and model complexity
            if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params <1000000):
                BatchFound= 64    
            if (os.cpu_count() <16) and (total_params <500000):
                BatchFound= 64  
            if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params <2000000) and (total_params >=1000000):
                BatchFound= 32      
            if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params >=2000000) and (total_params <10000000):
                BatchFound= 16  
            if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params >=10000000):
                BatchFound= 8       
            if (os.cpu_count() <16) and (total_params >5000000):
                BatchFound= 8    
            if total_params >100000000:
                BatchFound= 1

        except:
            pass
        try:

            #find percentage of memory used
            memoryused= psutil.virtual_memory()
            memoryused= float(str(memoryused).replace(" ", "").split("percent=")[1].split(",")[0])
            if memoryused >75.0:
                BatchFound= 8
            if memoryused >85.0:
                BatchFound= 4
            if memoryused >90.0:
                BatchFound= 2
            if total_params >100000000:
                BatchFound= 1
            print("Batch Size:  "+ str(BatchFound));    gc.collect()
        except:
            pass

        memoryused= [];    total_params= [];    GCPU= "";
        del memoryused, total_params, GCPU;    gc.collect()
        return BatchFound


