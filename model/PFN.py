'''
Structure:
The Particle Flow Network is made up of two NN's:
- Phi Network
- F Network
'''

from tqdm import tqdm
from Adam import *
import pandas as pd

cupy = False

try:
    import cupy as np
    import numpy as np1
    cupy = True
except ImportError:
    import numpy as np

class PFN:
    def __init__(self, phi_wei = None, phi_bias = None, f_wei = None, f_bias = None):

        self.phi_wei = [self.he_init(7, 100), self.he_init(100, 100), self.he_init(100, 128)] if phi_wei is None else [np.array(phi_wei[0]), np.array(phi_wei[1]), np.array(phi_wei[2])]
        self.phi_bias = [np.zeros(100), np.zeros(100), np.zeros(128)] if phi_bias is None else [np.array(phi_bias[0]), np.array(phi_bias[1]), np.array(phi_bias[2])]
        self.f_wei = [self.he_init(128, 100), self.he_init(100, 100), self.he_init(100, 2)] if f_wei is None else [np.array(f_wei[0]), np.array(f_wei[1]), np.array(f_wei[2])]
        self.f_bias = [np.zeros(100), np.zeros(100), np.zeros(2)] if f_bias is None else [np.array(f_bias[0]), np.array(f_bias[1]), np.array(f_bias[2])]

    def he_init(self, n_in, n_out):
        stddev = np.sqrt(2 / n_in)
        return np.random.randn(n_in, n_out) * stddev

    def one_hot(self, y):
        if cupy:
            y = y.get()
        y = pd.Series(y)
        y = pd.get_dummies(y).values.tolist()
        return np.array(y)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self, x):
        return np.where(x > 0, 1, 0)

    def loss(self, out, y):
        out = np.clip(out, 1e-8, 1 - 1e-8)  # Clip values to avoid log(0)
        y = np.array(y)
        return -np.sum(y * np.log(out)) / y.shape[0]

    def accuracy(self, y, preds):
        y = np.array(y)
        preds = np.array(preds)
        correct = np.sum(np.argmax(y, axis=1) == np.argmax(preds, axis=1))
        return correct / len(y)

    def backward(self, adam_phi, adam_f, x, y):
        feats = x.shape[0]
        num_particles = x.shape[1]
        m = y.shape[0]

        # Phi model
        phi_outputs = []
        all_as = []
        w_grads = [np.zeros_like(w) for w in self.phi_wei]
        b_grads = [np.zeros_like(b) for b in self.phi_bias]

        for i in range(feats):
            z1 = np.dot(x[i], self.phi_wei[0]) + self.phi_bias[0]
            a1 = self.ReLU(z1)

            z2 = np.dot(a1, self.phi_wei[1]) + self.phi_bias[1]
            a2 = self.ReLU(z2)

            z3 = np.dot(a2, self.phi_wei[2]) + self.phi_bias[2]
            a3 = self.ReLU(z3)

            phi_outputs.append(a3)
            all_as.append([a1, a2, a3])

        summed = np.sum(np.array(phi_outputs), axis=1)

        # F model
        z1 = np.dot(summed, self.f_wei[0]) + self.f_bias[0]
        a1 = self.ReLU(z1)

        z2 = np.dot(a1, self.f_wei[1]) + self.f_bias[1]
        a2 = self.ReLU(z2)

        z3 = np.dot(a2, self.f_wei[2]) + self.f_bias[2]
        a3 = self.softmax(z3)

        # Gradients for F model
        d3 = a3 - y
        d2 = np.dot(d3, self.f_wei[2].T) * self.dReLU(a2)
        d1 = np.dot(d2, self.f_wei[1].T) * self.dReLU(a1)

        dw3 = np.dot(a2.T, d3) / m
        dw2 = np.dot(a1.T, d2) / m
        dw1 = np.dot(summed.T, d1) / m

        db3 = np.sum(d3, axis=0) / m
        db2 = np.sum(d2, axis=0) / m
        db1 = np.sum(d1, axis=0) / m

        d_sum = np.dot(d1, self.f_wei[0].T)
        d_phi_output = np.tile(d_sum, (feats, 1))

        for i in range(feats):
            a1, a2, a3 = all_as[i]

            d3 = d_phi_output[i] * self.dReLU(a3)
            d2 = np.dot(d3, self.phi_wei[2].T) * self.dReLU(a2)
            d1 = np.dot(d2, self.phi_wei[1].T) * self.dReLU(a1)

            w_grads[2] += np.dot(a2.T, d3) / m
            w_grads[1] += np.dot(a1.T, d2) / m
            w_grads[0] += np.dot(x[i].T, d1) / m

            b_grads[2] += np.sum(d3, axis=0) / m
            b_grads[1] += np.sum(d2, axis=0) / m
            b_grads[0] += np.sum(d1, axis=0) / m

        updated_params = adam_f.update(self.f_wei + self.f_bias, [dw1, dw2, dw3, db1, db2, db3])
        self.f_wei = updated_params[:3]
        self.f_bias = updated_params[3:]

        updated_params_phi = adam_phi.update(self.phi_wei + self.phi_bias, w_grads + b_grads)
        self.phi_wei = updated_params_phi[:3]
        self.phi_bias = updated_params_phi[3:]

    def forward(self, x):

        feats = x.shape[0]
        num_particles = x.shape[1]

        # Phi model
        phi_outputs = []
        for i in range(feats):
            z1 = np.dot(x[i], self.phi_wei[0]) + self.phi_bias[0]
            a1 = self.ReLU(z1)

            z2 = np.dot(a1, self.phi_wei[1]) + self.phi_bias[1]
            a2 = self.ReLU(z2)

            z3 = np.dot(a2, self.phi_wei[2]) + self.phi_bias[2]
            a3 = self.ReLU(z3)

            phi_outputs.append(a3)
        
        summed = np.sum(np.array(phi_outputs), axis=1)

        # F model
        z1 = np.dot(summed, self.f_wei[0]) + self.f_bias[0]
        a1 = self.ReLU(z1)

        z2 = np.dot(a1, self.f_wei[1]) + self.f_bias[1]
        a2 = self.ReLU(z2)

        z3 = np.dot(a2, self.f_wei[2]) + self.f_bias[2]
        a3 = self.softmax(z3)

        return a3
    
    def predict(self, x):
        out = self.forward(np.array(x))
        if cupy == False:
            return out
        else:
            return np.asnumpy(out.get())

    def train(self, x, y, epoch = 10, batch_size = 1000, x_max = None):
        acc = []
        loss = []
        x = np.array(x)
        y = self.one_hot(y)
        print("Start Training: ")
        adam_f = AdamOptimizer()
        adam_phi = AdamOptimizer()

        for i in range(epoch):
            l = []
            preds = []
            if x_max is not None:
                
                indices = np.random.choice(x.shape[0], x_max, replace=False)
                x = x[indices]
                y = y[indices]


            for j in tqdm(range(0, len(x), batch_size)):
                end = j + batch_size
                data_batch = x[j:end]
                label_batch = y[j:end]

                out = self.forward(data_batch)
                preds.append(out)
                l.append(self.loss(out, label_batch))
                self.backward(adam_f, adam_phi, data_batch, label_batch)
            
            acc.append(self.accuracy(y, np.vstack(preds)))
            loss.append(sum(l)/len(x))
            print("epoch:", i, "loss:", loss[-1])

        y_preds = self.predict(x)
        print("Model Accuracy: ", self.accuracy(y, y_preds))
        return acc, loss, self.f_wei, self.f_bias, self.phi_wei, self.phi_bias