from __future__ import absolute_import, division, print_function, unicode_literals
from collections import deque
import numpy as np
import scipy.signal
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
import module.util as util
from copy import deepcopy

class MLP(object):
    def __init__(self, config, epochs=20, lr=1e-4, hdim=64, seed=0):
        self.seed = seed

        self.m = config["m"]
        self.J = config["J"]
        self.dt = config["dt"]
        self.stabilization_coeff = config["stabilization_coeff"]
        self.threshold = config["threshold"]
        
        self.w = config['w_init']
        self.epochs = epochs
        self.bound = config["bound"]
        self.lr = lr
        self.hdim = hdim
        tf.compat.v1.disable_eager_execution()
        
        self._build_graph2()
        self._build_dynamics()
        self.compile_model(self.w)

    def input_normalize(self, input_):
        res = np.array(input_ - (self.bound["max"] + self.bound["min"])/2.0)/(self.bound["max"] - self.bound["min"])*2.0
        if len(np.shape(res)) < 2:
            res = np.expand_dims(res,axis=0)
        return res

    def output_normalize(self, output_):
        return np.array(output_)/100.0

    def _build_graph2(self):
        model_input = keras.layers.Input(shape = (6,)) ## input should be normalized!!
        self.pos = model_input
        hlayer1 = keras.layers.Dense(self.hdim, activation=tf.nn.tanh)
        hlayer2 = keras.layers.Dense(self.hdim, activation=tf.nn.tanh)
        hlayer3 = keras.layers.Dense(self.hdim, activation=tf.nn.tanh)
        dropout1 = keras.layers.Dropout(0.25)
        fclayer = keras.layers.Dense(1, activation=tf.nn.sigmoid)
        outputbias = keras.layers.Lambda(lambda x : x-0.5)
        
        self.Cq = outputbias(fclayer(dropout1(hlayer3(hlayer2(hlayer1(self.pos))))))
        self.model = keras.models.Model(inputs = model_input, outputs = self.Cq)
               
        # hlayer1 = keras.layers.Dense(self.hdim, activation=tf.nn.tanh)(model_input)
        # hlayer2 = keras.layers.Dense(self.hdim, activation=tf.nn.tanh)(hlayer1)
        # hlayer3 = keras.layers.Dense(self.hdim, activation=tf.nn.tanh)(hlayer2)
        # dropout1 = keras.layers.Dropout(0.25)(hlayer3)
        # hlayer3 = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout1)
        # self.Cq = keras.layers.Lambda(lambda x : x-0.5)(hlayer3)
        # self.model = keras.models.Model(inputs = model_input, outputs = self.Cq)
    
        # summarize layers
        print(self.model.summary())
        self.predict_func = keras.backend.function(self.pos, self.model.output)

    def _build_dynamics(self):
        self.grad_C = keras.layers.Lambda( lambda z: K.gradients( z[ 0 ][:,0], z[ 1 ] ) )( [ self.model.output, self.pos ] )
        grad_C_sq = tf.squeeze(self.grad_C)
        self.H1 = keras.layers.Lambda( lambda z: K.gradients( z[0][:,0], z[1] ) )( [ grad_C_sq, self.pos ] )
        self.H2 = keras.layers.Lambda( lambda z: K.gradients( z[0][:,1], z[1] ) )( [ grad_C_sq, self.pos ] )
        self.H3 = keras.layers.Lambda( lambda z: K.gradients( z[0][:,2], z[1] ) )( [ grad_C_sq, self.pos ] )
        self.H4 = keras.layers.Lambda( lambda z: K.gradients( z[0][:,3], z[1] ) )( [ grad_C_sq, self.pos ] )
        self.H5 = keras.layers.Lambda( lambda z: K.gradients( z[0][:,4], z[1] ) )( [ grad_C_sq, self.pos ] )
        self.H6 = keras.layers.Lambda( lambda z: K.gradients( z[0][:,5], z[1] ) )( [ grad_C_sq, self.pos ] )
        
        # build function
        # self.Cq_cal = keras.backend.function(self.input, self.Cq)
        self.grad_c_cal = keras.backend.function(self.pos, self.grad_C)
        
        

    def compile_model(self, w):
        self.w = w
        print("weights : {}".format(w))
        
        def K_inner_product(A,B):
            return K.sum(A * B, axis=1, keepdims=True)

        def mse_loss(y_true, y_pred):
            mse_loss = keras.losses.mse(y_true, y_pred)
            return w[0]*mse_loss
        
        def grad_norm_loss(y_true, y_pred):
            return w[1]*K.mean(K.square(self.grad_C))
        
        def Hess_loss(y_true, y_pred):
            Hess_loss = K.mean( K.square(self.H1) + K.square(self.H2) + K.square(self.H3) + K.square(self.H4) + K.square(self.H5) + K.square(self.H6) )
            return w[2]*Hess_loss
        
        def l2dist(x,y):
            return K.sqrt(K.sum((x-y)**2, axis=1, keepdims=True))
        
        def tot_loss(y_true, y_pred):
            return mse_loss(y_true, y_pred) + grad_norm_loss(y_true, y_pred);# + Hess_loss(y_true, y_pred)
        
        self.model.compile(optimizer='adam',
              loss=tot_loss,
            #   metrics=[mse_loss, grad_norm_loss, Hess_loss])
            metrics=[mse_loss, grad_norm_loss])
        
    def fit(self, x, y, epochs=300, batch_size=256):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        history = self.model.fit(x,
                    y,
                    shuffle=True,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=1)
        return 0

    def predict(self, pos):
        return np.squeeze(self.predict_func(self.input_normalize(pos)))

    def grad(self, pos):
        return np.squeeze(self.grad_c_cal(self.input_normalize(pos)))/(self.bound["max"] - self.bound["min"])*2.0

    def load_model(self, load_dir):
       
        # from tensorflow.keras.models import model_from_json
        # json_file = open(load_dir+"/model"+str(nepochs)+".json", "r")
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        self.model.load_weights(load_dir)
        print("=====================")
        print("Loaded model from disk")
        
        
    def txt_wright(self, filename, A):
        if A.ndim == 1:
            B = np.expand_dims(A,axis=1)
        else :
            B = deepcopy(A)       
    
        f = open(filename, 'w')
        for i in range(B.shape[0]):
            data = ''
            for j in range(B.shape[1]):
                data += str(B[i,j])
                if j!=B.shape[1]-1:
                    data += ' '
            f.write(data + "\n")
        f.close()

    def export_weight(self):
                
        # 3 layer
        W1 = self.model.layers[1].get_weights()[0]
        b1 = self.model.layers[1].get_weights()[1]
        W2 = self.model.layers[2].get_weights()[0]
        b2 = self.model.layers[2].get_weights()[1]
        W3 = self.model.layers[3].get_weights()[0]
        b3 = self.model.layers[3].get_weights()[1]
        W4 = self.model.layers[5].get_weights()[0]
        b4 = self.model.layers[5].get_weights()[1]
        
        self.txt_wright("weight/W1.txt", W1)
        self.txt_wright("weight/b1.txt", b1)
        self.txt_wright("weight/W2.txt", W2)
        self.txt_wright("weight/b2.txt", b2)
        self.txt_wright("weight/W3.txt", W3)
        self.txt_wright("weight/b3.txt", b3)
        self.txt_wright("weight/W4.txt", W4)
        self.txt_wright("weight/b4.txt", b4)

        print("finish save")
        
    def load_weights_to_dyn(self, dyn):
        W1 = np.squeeze(self.model.layers[1].get_weights()[0]).astype('float64')
        b1 = np.squeeze(self.model.layers[1].get_weights()[1]).astype('float64')
        W2 = np.squeeze(self.model.layers[2].get_weights()[0]).astype('float64')
        b2 = np.squeeze(self.model.layers[2].get_weights()[1]).astype('float64')
        W3 = np.squeeze(self.model.layers[3].get_weights()[0]).astype('float64')
        b3 = np.squeeze(self.model.layers[3].get_weights()[1]).astype('float64')
        W4 = np.squeeze(self.model.layers[5].get_weights()[0]).astype('float64')
        b4 = np.array([np.squeeze(self.model.layers[5].get_weights()[1]).astype('float64')])
        dyn.change_weight(W1, W2, W3, W4,b1,b2, b3, b4)