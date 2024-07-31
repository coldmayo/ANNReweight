from tqdm import tqdm
import pandas as pd
import random
from Adam import *
#from utils import *

cupy = False

try:
    import cupy as np
    import numpy as np1
    cupy = True
except ImportError:
    import numpy as np

class FNN:
    def __init__(self, input_size, hidden_size, num_class, w1 = None, w2 = None, w3 = None, w4 = None, b1 = None, b2 = None, b3 = None, b4 = None):

        self.input_size = input_size

        # set weights
        self.w1 = self.he_init(input_size, hidden_size) if w1 is None else np.array(w1)
        self.w2 = self.he_init(hidden_size, hidden_size) if w2 is None else np.array(w2)
        self.w3 = self.he_init(hidden_size, hidden_size) if w3 is None else np.array(w3)
        self.w4 = self.he_init(hidden_size, num_class) if w4 is None else np.array(w4)

        # set biases
        self.b1 = np.zeros(input_size) if b1 is None else np.array(b1)
        self.b2 = np.zeros(hidden_size) if b2 is None else np.array(b2)
        self.b3 = np.zeros(hidden_size) if b3 is None else np.array(b3)
        self.b4 = np.zeros(num_class) if b4 is None else np.array(b4)

    def he_init(self, n_in, n_out):
        stddev = np.sqrt(2 / n_in)
        return np.random.randn(n_in, n_out) * stddev

    def one_hot(self, y):
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
    
    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.ReLU(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.ReLU(z2)

        z3 = np.dot(a2, self.w3) + self.b3
        a3 = self.ReLU(z3)

        z4 = np.dot(a3, self.w4) + self.b4
        a4 = self.softmax(z4)

        return a4

    def backward(self, adam, x, y):
        m = y.shape[0]

        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.ReLU(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.ReLU(z2)

        z3 = np.dot(a2, self.w3) + self.b3
        a3 = self.ReLU(z3)

        z4 = np.dot(a3, self.w4) + self.b4
        a4 = self.softmax(z4)

        d4 = a4 - y
        d3 = np.dot(d4, self.w4.T) * self.dReLU(a3)
        d2 = np.dot(d3, self.w3.T) * self.dReLU(a2)
        d1 = np.dot(d2, self.w2.T) * self.dReLU(a1)

        dw4 = np.dot(a3.T, d4) / m
        dw3 = np.dot(a2.T, d3) / m
        dw2 = np.dot(a1.T, d2) / m
        dw1 = np.dot(x.T, d1) / m

        db4 = np.sum(d4, axis=0) / m
        db3 = np.sum(d3, axis=0) / m
        db2 = np.sum(d2, axis=0) / m
        db1 = np.sum(d1, axis=0) / m
        updated_params = adam.update([self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4], [dw1, dw2, dw3, dw4, db1, db2, db3, db4])
        self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4 = updated_params

    def loss(self, out, y):
        out = np.clip(out, 1e-8, 1 - 1e-8)  # Clip values to avoid log(0)
        y = np.array(y)
        return -np.sum(y * np.log(out)) / y.shape[0]

    def accuracy(self, y, preds):
        y = np.array(y)
        preds = np.array(preds)
        correct = np.sum(np.argmax(y, axis=1) == np.argmax(preds, axis=1))
        return correct / len(y)

    def train(self, x, y, epoch = 10, batch_size = 1000, x_max = None):
        acc = []
        loss = []
        x = np.array(x)
        y = self.one_hot(y)
        print("Start Training: ")
        adam = AdamOptimizer()
        for i in range(epoch):
            l = []
            preds = []
            if x_max is not None:
                x_old = x
                y_old = y

                x = []
                y = []

                for k in np.random.choice(len(x_old), x_max):
                    x.append(x_old[k])
                    y.append(y_old[k])

                x = np.array(x)
                x = x.reshape((x_max, self.input_size))
                y = np.array(y)

            for j in tqdm(range(0, len(x), batch_size)):
                end = j + batch_size
                data_batch = x[j:end]
                label_batch = y[j:end]

                out = self.forward(data_batch)
                preds.append(out)
                l.append(self.loss(out, label_batch))
                self.backward(adam, data_batch, label_batch)

            acc.append(self.accuracy(y, np.vstack(preds)))
            loss.append(sum(l)/len(x))
            print("epoch:", i, "loss:", loss[-1])
        y_preds = self.predict(x)
        print("Model Accuracy: ", self.accuracy(y, y_preds))
        if cupy == False:
            return acc, loss, self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4
        else:
            loss = np.array(loss)
            return acc, np.asnumpy(loss.get()), self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4

    def predict(self, x):
        out = self.forward(np.array(x))
        if cupy == False:
            return out
        else:
            return np.asnumpy(out.get())
