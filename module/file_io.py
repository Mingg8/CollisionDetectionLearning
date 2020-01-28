import numpy as np
from scipy.io import loadmat

class FileIO:
    def dataLoader(filename, input, output):
        mat_contents = loadmat(filename)
        i_data = mat_contents[input]
        o_data = mat_contents[output]
        return i_data, o_data

    def loadFile(model_save_dir, m_file_name, w_file_name):
        # load model
        try:
            json_file = open(model_save_dir + m_file_name, 'r')
        except:
            print("json file failed to load, dir: "
                + model_save_dir + m_file_name)
            sys.exit(-1)

        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weight
        loaded_model.load_weights(model_save_dir + w_file_name)
        print("Loaded model from disk, name: "
            + model_save_dir + m_file_name)
        return loaded_model

    def saveData(model_save_dir, file_name, real_i, real_o,
        pred_o, pred_g, pred_g2):
        print(np.shape(real_i))
        print(np.shape(real_o))
        length = np.shape(real_i)[0]
        real_o = np.reshape(real_o, (length, 1))
        pred_o = np.reshape(pred_o, (length, 1))

        mat = np.append(real_i, real_o, axis = 1)
        mat = np.append(mat, pred_o, axis = 1)
        mat = np.append(mat, pred_g, axis = 1)
        mat = np.append(mat, pred_g2, axis = 1)
        np.savetxt(model_save_dir + file_name, mat)
        