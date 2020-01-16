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


def dataLoader(filename):
    mat_contents = loadmat(filename)
    i_data = mat_contents['input']
    o_data = mat_contents['output']
    return i_data, o_data

def dataPlot(prediction, output):
    sorted_ind = sorted(range(len(output)), key = lambda k:output[k])
    sorted_prediction = [prediction[i] for i in sorted_ind]
    sorted_output = [output[i] for i in sorted_ind]
    x = range(len(output))

    plt.plot(x, sorted_prediction, 'b', sorted_output, 'r')
    plt.show()


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
    def __init__(self, m_file_name = "", w_file_name = ""):
        if (m_file_name == "" and w_file_name == ""):
            self.model = Sequential()
            # training
            self.model.add(Dense(250, input_dim = 3, activation = 'relu'))
            self.model.add(Dense(250, activation = 'relu'))
            self.model.add(Dense(250, activation = 'relu'))
            self.model.add(Dense(1, activation = 'tanh'))

        else:
            self.model = self.loadFile(m_file_name, w_file_name)

    def train(self, filename, save):
        print(self.model)
        i_data, o_data = dataLoader(filename)
        i_data = i_data[1:100, :]
        o_data = o_data[1:100]

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
        self.model.fit(train_input, train_output, epochs = 50,
            batch_size = 100,
            validation_data = (valid_input, valid_output)
            )
        
        test_input, test_output = n.dataNormalize(test_i, test_o)
        accuracy = self.model.evaluate(test_input, test_output)
        print('Accuracy: %.2f' % (accuracy * 100))

        self.evaluation(test_input, test_output, n)

    def loadFile(self, m_file_name, w_file_name):
        # load model
        json_file = open(m_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weight
        loaded_model.load_weights(w_file_name)
        print("Loaded model from disk")
        return loaded_model
    
    def saveFile(self, m_file_name, w_file_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(m_file_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(w_file_name)
        print("Save model to disk")

    def evaluation(self, i_data, o_data, n):
        # predict
        predictions = self.model.predict(i_data)
        predict_output = n.dataUnnormalize(predictions)
        real_output = n.dataUnnormalize(o_data)
        # round predictions
        # eval = np.linalg.norm(predict_output - real_output)
        error = abs(predict_output - real_output)
        print("max error: %.10f" %(np.max(error)))
        dataPlot(predict_output, real_output)


    def loadAndTest(self, file_path):
        filename = 'obj/data/data_final.mat'
        i_data, o_data = dataLoader(file_path + filename)
        train_i, train_o, test_i, test_o = trainTestSplit(i_data, o_data)
        n = Normalize(test_i, test_o)
        test_input, test_output = n.dataNormalize(test_i, test_o)

        self.model = loadFile(file_path + 'model.json', file_path + 'model.h5')
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam',
            metric = ['accuracy'])
        self.evaluation(test_input, test_output, n)

if __name__ == "__main__":
    file_path = '/home/mjlee/workspace/NutLearning/'
    # loadAndTest(file_path)
    t = Train()
    t.train(file_path + 'obj/data/data_final.mat', 0)
    t.saveFile("model.json", "model.h5")