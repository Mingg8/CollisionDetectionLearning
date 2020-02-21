import numpy as np
from sklearn import svm
from module.file_io import FileIO
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

i_data, o_data = FileIO.dataLoader(
    '/home/mjlee/workspace/CollisionDetectionLearning/obj/data/total_data_only_penet_fine_data3_edit.mat',
    'input_data',
    'output_data')

i_data_train = np.transpose(np.array(i_data))[:900000]
o_data_train = np.transpose(np.array(o_data))[:900000]
i_data_test = np.transpose(np.array(i_data))[900000:]
o_data_test = np.transpose(np.array(o_data))[900000:]
print(np.shape(i_data_test))
print(np.shape(o_data_train))

C = 0
gamma = 20

rbf_svc = svm.SVC(kernel = 'rbf', gamma = gamma, coef0 = C, verbose = True)
rbf_svc.fit(i_data_train[:], o_data_train[:])

prediction = rbf_svc.predict(i_data_test)
print("accuracy: {}".format(accuracy_score(o_data_test, prediction)))


joblib_file = "joblib_model.pkl"
joblib.dump(rbf_svc, joblib_file)

