import numpy as np
from scipy.io import loadmat
import numpy.linalg as LA
from math import sqrt, floor
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense

from module.normalize import Normalize
from datetime import datetime


def dataLoader(filename, input, output):
    mat_contents = loadmat(filename)
    i_data = mat_contents[input]
    o_data = mat_contents[output]
    return i_data, o_data


def trainValidTestSplit(i_data, o_data, tr, va, te):
    train_i = []
    train_o = []
    test_i = []
    test_o = []
    valid_i = []
    valid_o = []

    for i in range(len(o_data)):
        if i % 100 < int(tr * 100) :
            train_i.append(i_data[i,:])
            train_o.append(o_data[i])
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

class Train:
    def __init__(self, err, m_file_name = "", w_file_name = ""):
        self.error_bound = err
        if (m_file_name == "" and w_file_name == ""):
            self.model = Sequential()
            # training
            self.model.add(Dense(250, input_dim = 3, activation = 'relu'))
            self.model.add(Dense(250, activation = 'relu'))
            self.model.add(Dense(250, activation = 'relu'))
            self.model.add(Dense(1, activation = 'tanh'))

        else:
            self.model = self.loadFile(m_file_name, w_file_name)

    def train(self, i_data, o_data, save):
        print(self.model)
        train_i, train_o, valid_i ,valid_o, test_i, test_o = \
            trainValidTestSplit(i_data, o_data, 0.7, 0.15, 0.15)
        # scaler = StandardScaler()
        # scaler.fit(train_o)
        # train_input = scaler.transform(train_i)

        n = Normalize(train_i, train_o)
        train_input, train_output = n.dataNormalize(train_i, train_o)
        valid_input, valid_output = n.dataNormalize(valid_i, valid_o)

        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam',
            metric = ['accuracy'])
        self.model.fit(train_input, train_output, epochs = 30,
            batch_size = 100,
            validation_data = (valid_input, valid_output)
            )
        
        test_input, test_output = n.dataNormalize(test_i, test_o)

        min, max, left_i, right_i = self.evaluation(test_input, test_output, n)
        return min, max, left_i ,right_i

    def loadFile(self, m_file_name, w_file_name):
        # load model
        json_file = open(m_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weight
        loaded_model.load_weights(w_file_name)
        print("Loaded model from disk, name: " + m_file_name)
        return loaded_model
    
    def saveFile(self, m_file_name, w_file_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(m_file_name, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(w_file_name)
        print("Save model to disk, name: " + m_file_name)

    def evaluation(self, i_data, o_data, n):
        # predict
        prediction = self.model.predict(i_data)
        predict_output = n.dataUnnormalize(prediction)
        real_output = n.dataUnnormalize(o_data)

        sorted_ind = sorted(range(len(real_output)), key = lambda k:real_output[k])
        sorted_prediction = np.array([predict_output[i] for i in sorted_ind])
        sorted_output = np.array([real_output[i] for i in sorted_ind])

        error = abs(sorted_prediction - sorted_output)

        print("max error: %.10f" %(np.max(error)))

        x = range(len(sorted_output))
        plt.plot(x, sorted_prediction, 'b', sorted_output, 'r')
        plt.savefig('figure/figure_'+str(datetime.now())+'.png')
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
        return sorted_output[left_i], sorted_output[right_i], left_i, right_i

    def loadAndTest(self, file_path):
        filename = 'obj/data/data_final.mat'
        i_data, o_data = dataLoader(file_path + filename, input, output)
        train_i, train_o, test_i, test_o = trainTestSplit(i_data, o_data)
        n = Normalize(test_i, test_o)
        test_input, test_output = n.dataNormalize(test_i, test_o)

        self.model = loadFile(file_path + 'model.json', file_path + 'model.h5')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam',
            metric = ['accuracy'])
        self.evaluation(test_input, test_output, n)

if __name__ == "__main__":
    data_num = 300000
    file_path = '/home/mjlee/workspace/NutLearning/'
    file_name = [
        'obj/data/data_final1.mat',
        'obj/data/data_final2.mat',
        'obj/data/data_final3.mat',
        'obj/data/data_final4.mat',
        'obj/data/data_final5.mat']

    add_ratio = 0.7
    error_bound = 0.002
    itr_num = 4

    # loadAndTest(file_path)
    t = Train(error_bound)
    i_data, o_data = dataLoader(file_path + file_name[0],
        'input',
        'output') # 1000k

    i_data = i_data[1:data_num, :]
    o_data = o_data[1:data_num, :]

    sorted_ind = sorted(range(len(o_data)), key = lambda k:o_data[k])
    sorted_input = np.array([i_data[i] for i in sorted_ind])
    sorted_output = np.array(sorted(o_data))

    a, b, left_i, right_i = t.train(sorted_input, sorted_output, False)
    t.saveFile("model0.json", "model0.h5")
    del t

    for itr in range(1, len(file_name)):
        t = Train(error_bound, "model" + str(itr-1) + ".json",
            "model" + str(itr-1) + ".h5")
        new_i, new_o = dataLoader(file_path + file_name[itr],
            'input',
            'output') # 1000k

        new_i = new_i[1:data_num, :]
        new_o = new_o[1:data_num, :]
        change_num = 0
        for i in range(len(new_o)):
            if (new_o[i] < a or new_o[i] > b):
                index = left_i + int((right_i - left_i) * np.random.rand(1))
                sorted_input[index] = new_i[i]
                sorted_output[index] = new_o[i]
                change_num += 1

        print("changing: {}".format(change_num))
        del new_i, new_o

        sorted_ind = sorted(range(len(sorted_output)), key = lambda k:sorted_output[k])
        sorted_input = np.array([sorted_input[i] for i in sorted_ind])
        sorted_output = np.array(sorted(sorted_output))
        
        a, b, left_i, right_i = t.train(sorted_input, sorted_output, False)
        t.saveFile('model'+str(itr) +'.json', "model"+str(itr)+".h5")
        del t
