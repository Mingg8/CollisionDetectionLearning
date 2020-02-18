import numpy as np
from config import config

class Normalize():
    def __init__(self, i_data, o_data, save_dir):
        max_i = np.amax(i_data, axis = 0)
        min_i = np.amin(i_data, axis = 0)
        max_o = np.amax(o_data, axis = 0)
        min_o = np.amin(o_data, axis = 0)

        self.a_i = 2 / (max_i - min_i)
        self.b_i = -(min_i + max_i) / (max_i - min_i)
        self.a_o = 2 / (max_o - min_o)
        self.b_o = -(min_o + max_o) / (max_o - min_o)

        self.saveCoeffs(save_dir)

    def dataNormalize(self, i_data, o_data):
        for l in range(config['input_num']):
            i_data[:, l] = self.a_i[l] * i_data[:, l] + self.b_i[l]
        for l in range(config['output_num']):
            o_data[:, l] = self.a_o[l] * o_data[:, l] + self.b_o[l]
        return i_data, o_data

    def oDataUnnormalize(self, o_data):
        return (o_data - self.b_o) / self.a_o
    
    def iDataUnnormalize(self, i_data):
        return (i_data - self.b_i) / self.a_i

    def gDataUnnormalize(self, g_data):
        return (g_data - self.b_o[1:]) / self.a_o[1:]

    def saveCoeffs(self, save_dir):
        input_coeff = np.append(self.a_i, self.b_i)
        output_coeff = np.append(self.a_o, self.b_o)
        np.savetxt(save_dir + '/input_coeff.csv', input_coeff)
        np.savetxt(save_dir + '/output_coeff.csv', output_coeff)
