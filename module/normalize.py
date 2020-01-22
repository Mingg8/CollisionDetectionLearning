import numpy as np

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
        for l in range(4):
            o_data[:, l] = self.a_o[l] * o_data[:, l] + self.b_o[l]
        return i_data, o_data

    def dataUnnormalize(self, o_data):
        return (o_data - self.b_o) / self.a_o