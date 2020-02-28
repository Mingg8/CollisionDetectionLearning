import numpy as np
from module.normalize import Normalize
from config.config import config

class DataProcessing:
    def __init__(self, i_data, o_data, data_num, save_dir):
        self.data_num = data_num

        # ob = config["output_bound"]
        # i_data, o_data = self.preprocessing(i_data, o_data, ob)

        sr = config["trainset_ratio"]
        train_i, train_o, test_i, test_o = \
            self.dataSplit(i_data, o_data, sr)
        
        self.n = Normalize(train_i, train_o, save_dir)
        self.train_input, self.train_output = \
            self.n.dataNormalize(train_i, train_o)
        self.test_input, self.test_output = \
            self.n.dataNormalize(test_i, test_o)
        
    def dataSplit(self, i_data, o_data, tr = 0.85):
        data = np.append(i_data, o_data, axis = 1)
        np.random.shuffle(data)

        train_i = []
        train_o = []
        test_i = []
        test_o = []

        for i in range(self.data_num):
            if i < int(self.data_num * tr):
                train_i.append(data[i, 0:config['input_num']])
                train_o.append(data[i, config['input_num']:])
            else:
                test_i.append(data[i, 0:config['input_num']])
                test_o.append(data[i, config['input_num']:])

        train_i = np.array(train_i)
        train_o = np.array(train_o)
        test_i = np.array(test_i)
        test_o = np.array(test_o)

        return train_i, train_o, test_i, test_o

    def preprocessing(self, i_data, o_data, bound):
        # sort data
        sorted_ind = sorted(range(len(o_data)), key = lambda k:o_data[k][0])
        sorted_i_data = np.array([i_data[i] for i in sorted_ind])
        sorted_o_data = np.array([o_data[i] for i in sorted_ind])

        # cut data
        sorted_input = []
        sorted_output = []
        for i in range(len(o_data)):
            if sorted_o_data[i] > bound[0]:
                if sorted_o_data[i] > bound[1]:
                    break
                sorted_input.append(sorted_i_data[i])
                sorted_output.append(sorted_o_data[i])
        sorted_input = np.array(sorted_input)
        sorted_output = np.array(sorted_output)
        return sorted_input, sorted_output
