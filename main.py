import numpy as np
from scipy.io import loadmat
import numpy.linalg as LA
from math import sqrt, floor


from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

class Normalize():
    def __init__(self, i_data, o_data):
        max_i = np.amax(i_data, axis = 0)
        min_i = np.amin(i_data, axis = 0)
        max_o = np.amax(o_data, axis = 0)
        min_o = np.amin(o_data, axis = 0)

        self.a_i = 2 / (max_i - min_i)
        self.b_i = -(min_i + max_i) / (max_i - min_i)
        self.a_o = 2 / (max_o - min_o)
        self.b_o = -(min_o + max_o) / (max_o - min_o)

    def dataNormalize(self, i_data, o_data):
        for l in range(3):
            i_data[:, l] = self.a_i[l] * i_data[:, l] + self.b_i[l]
        o_data = self.a_o * o_data + self.b_o
        return i_data, o_data

    def dataUnnormalize(self, o_data):
        return (o_data - self.b_o) / self.a_o

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


def evaluation(model, i_data, o_data, n):
    # predict
    predictions = model.predict(i_data)
    predict_output = n.dataUnnormalize(predictions)
    real_output = n.dataUnnormalize(o_data)
    # round predictions
    # eval = np.linalg.norm(predict_output - real_output)
    error = abs(predict_output - real_output)
    print("max error: %.10f" %(np.max(error)))
    dataPlot(predict_output, real_output)



def saveFile(model, m_file_name, w_file_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(m_file_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(w_file_name)
    print("Save model to disk")

def loadFile(m_file_name, w_file_name):
    # load model
    json_file = open(m_file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weight
    loaded_model.load_weights(w_file_name)
    print("Loaded model from disk")
    return loaded_model

def trainTestSplit(i_data, o_data):
    train_i = []
    train_o = []
    test_i = []
    test_o = []
    for i in range(len(o_data)):
        if i % 10 <= 2 :
            test_i.append(i_data[i, :])
            test_o.append(o_data[i, :])
        else :
            train_i.append(i_data[i,:])
            train_o.append(o_data[i])

    train_i = np.array(train_i)
    train_o = np.array(train_o)
    test_i = np.array(test_i)
    test_o = np.array(test_o)
    return train_i, train_o, test_i, test_o

def train(file_path):
    filename = 'obj/data/final_data.mat'
    i_data, o_data = dataLoader(file_path + filename)

    train_i, train_o, test_i, test_o = trainTestSplit(i_data, o_data)
    n = Normalize(train_i, train_o)
    train_input, train_output = n.dataNormalize(train_i, train_o)

    # train_input = train_input[1:100, :]
    # train_output = train_output[1:100, :]

    # training
    model = Sequential()
    model.add(Dense(100, input_dim = 3, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation = 'tanh'))
    # model.add(Dense(1))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam',
        metric = ['accuracy'])
    try:
        model.fit(train_input, train_output, epochs = 80, batch_size = 100)
    except KeyboardInterrupt:
        print("a")
    
    test_input, test_output = n.dataNormalize(test_i, test_o)
    accuracy = model.evaluate(test_input, test_output)
    print('Accuracy: %.2f' % (accuracy * 100))

    saveFile(model, "model.json", "model.h5")

    evaluation(model, test_input, test_output, n)

def loadAndTest(file_path):
    filename = 'obj/data/final_data.mat'
    i_data, o_data = dataLoader(file_path + filename)
    train_i, train_o, test_i, test_o = trainTestSplit(i_data, o_data)
    n = Normalize(test_i, test_o)
    test_input, test_output = n.dataNormalize(test_i, test_o)

    model = loadFile(file_path + 'model.json', file_path + 'model.h5')
    model.compile(loss = 'mean_squared_error', optimizer = 'adam',
        metric = ['accuracy'])
    evaluation(model, test_input, test_output, n)

if __name__ == "__main__":
    file_path = '/home/mjlee/workspace/NutLearning/'
    # loadAndTest(file_path)
    train(file_path)