from tqdm import tqdm
import pandas as pd
import random

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
        if w1 is None:
            self.w1 = np.random.rand(input_size, hidden_size)
        else:
            self.w1 = np.array(w1)

        if w2 is None:
            self.w2 = np.random.rand(hidden_size, hidden_size)
        else:
            self.w2 = np.array(w2)

        if w3 is None:
            self.w3 = np.random.rand(hidden_size, hidden_size)
        else:
            self.w3 = np.array(w3)

        if w4 is None:
            self.w4 = np.random.rand(hidden_size, num_class)
        else:
            self.w4 = np.array(w4)

        # set biases
        self.b1 = np.ones(hidden_size) if b1 is None else np.array(b1)
        self.b2 = np.ones(hidden_size) if b2 is None else np.array(b2)
        self.b3 = np.ones(hidden_size) if b3 is None else np.array(b3)
        self.b4 = np.ones(num_class) if b4 is None else np.array(b4)

    def one_hot(self, y):
        y = pd.Series(y)
        y = pd.get_dummies(y).values.tolist()
        return np.array(y)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.ReLU(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.ReLU(z2)

        z3 = np.dot(a2, self.w2) + self.b3
        a3 = self.ReLU(z3)

        z4 = np.dot(a3, self.w4) + self.b4
        a4 = self.sigmoid(z4)

        return a4

    def backward(self, x, y, alpha):

        m = y.shape[0]
        
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.ReLU(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.ReLU(z2)

        z3 = np.dot(a2, self.w3) + self.b3
        a3 = self.ReLU(z3)

        z4 = np.dot(a3, self.w4) + self.b4
        a4 = self.sigmoid(z4)

        d4 = a4-y
        d3 = np.dot(d4, self.w4.T) * self.dReLU(a3)
        d2 = np.dot(d3, self.w3.T) * self.dReLU(a2)
        d1 = np.dot(d2, self.w2.T) * self.dReLU(a1)
        
        w1_alt = np.dot(x.T, d1) / m
        w2_alt = np.dot(a1.T, d2) / m
        w3_alt = np.dot(a2.T, d3) / m
        w4_alt = np.dot(a3.T, d4) / m

        b1_alt = np.sum(d1, axis=0) / m
        b2_alt = np.sum(d2, axis=0) / m
        b3_alt = np.sum(d3, axis=0) / m
        b4_alt = np.sum(d4, axis=0) / m

        self.w1 -= alpha*w1_alt
        self.w2 -= alpha*w2_alt
        self.w3 -= alpha*w3_alt
        self.w4 -= alpha*w4_alt

        self.b1 -= alpha*b1_alt
        self.b2 -= alpha*b2_alt
        self.b3 -= alpha*b3_alt
        self.b4 -= alpha*b4_alt

    def loss(self, out, y):
        out = np.array(out)
        y = np.array(y)
        return -np.sum(y * np.log(out)) / y.shape[0]

    def accuracy(self, y, preds):
        y = np.array(y)
        preds = np.array(preds)
        correct = np.sum(np.argmax(y, axis=1) == np.argmax(preds, axis=1))
        return correct / len(y)

    def train(self, x, y, alpha = 0.01, epoch = 10, batch_size = 1000, x_max = None):
        acc = []
        loss = []
        x = np.array(x)
        y = self.one_hot(y)
        print("Start Training: ")
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
                self.backward(data_batch, label_batch, alpha)
            
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
