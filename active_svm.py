import os
import numpy as np
from datetime import datetime
import time

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from config.config_svm import config
from module.file_io import FileIO
from module.svm_modules import EntropySelection

file_path = str(os.getcwd())
now = datetime.now()
now_string = now.strftime("%Y-%m-%d_%H:%M")
save_directory = file_path + '/old_results/svm/' + now_string
try:
    os.makedirs(save_directory)
except:
    print("already exists")

i_data, o_data = FileIO.dataLoader(
        config['data_file_name'],
        config['input_file'],
        config['output_file'])
i_pool = np.transpose(np.array(i_data))
o_pool = np.transpose(np.array(o_data))

i_pool = i_pool[:300000]
o_pool = o_pool[:300000]
del i_data, o_data

# new = True
new = False
filename = '/home/mjlee/workspace/CollisionDetectionLearning/model_2020-02-21_17:38.pkl'
# find initial set
test_set = np.random.choice(np.shape(i_pool)[0],
        config['n_queries'],
        replace = False
        )
i_test = i_pool[test_set]
o_test = o_pool[test_set]
i_pool, o_pool = np.delete(i_pool, test_set, axis = 0), \
                np.delete(o_pool, test_set, axis = 0)

max_n = int(np.shape(i_pool)[0] / config["sample_num"])
if (config["n_queries"] > max_n):
    iter_num = max_n
else:
    iter_num = config["n_queries"]
    

permutation = np.random.choice(np.shape(i_pool)[0],
        config['sample_num'],
        replace = False
        )
i_train = i_pool[permutation]
o_train = o_pool[permutation]
i_pool, o_pool = np.delete(i_pool, permutation, axis = 0), \
                        np.delete(o_pool, permutation, axis = 0)

for index in range(iter_num):
    if index == 0:
        if new:
            # rbf_svc = RandomForestClassifier(min_samples_leaf = 20)
            rbf_svc = svm.SVC(kernel = 'rbf',
                    gamma = config["gamma"],
                    coef0 = config["C"],
                    probability = True,
                    cahce_size = 7000
                    )
        else:
            rbf_svc = joblib.load(filename)
    else:
        start = time.time()
        probas_val = rbf_svc.predict_proba(i_pool)
        uncertain_samples = EntropySelection(probas_val[:config['val_num']], config['sample_num'])

        uncertain_samples = np.reshape(uncertain_samples, (np.shape(uncertain_samples)[0], ))
#         print(np.shape(i_train), np.shape(i_pool[uncertain_samples]))
#         i_train = np.concatenate((i_train, i_pool[uncertain_samples]), axis = 0)
#         uncertain_samples = np.reshape(uncertain_samples, (np.shape(uncertain_samples)[0], ))
#         o_train = np.reshape(o_train, (np.shape(o_train)))
#         o_train = np.concatenate((np.squeeze(o_train), np.squeeze(o_pool[uncertain_samples])), axis = 0)
        i_train = i_pool[uncertain_samples]
        o_train = o_pool[uncertain_samples]

        i_pool, o_pool = np.delete(i_pool, uncertain_samples, axis = 0) \
            , np.delete(o_pool, uncertain_samples)
        end = time.time()
        print("Elapsed time (data querying): {}".format(end - start))

        start = time.time()
        rbf_svc = joblib.load(filename)
        end = time.time()
        print("Elapsed time (loading): {}".format(end - start))

    start = time.time()
    rbf_svc.fit(i_train, o_train)
    end = time.time()
    print("Elapsed time (training): {}".format(end - start))

    prediction = rbf_svc.predict(i_pool)
    print("accuracy: {}".format(accuracy_score(o_pool, prediction)))

    now = datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H:%M")
    filename = save_directory + "/model_" + now_string + ".pkl"
    joblib.dump(rbf_svc, filename)
    new = False
