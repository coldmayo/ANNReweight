import numpy as np
from tqdm import tqdm
import pandas as pd

class FNN:
    def __init__(self, input_size, hidden_size, num_class, w1 = None, w2 = None, b1 = None, b2 = None):
        # set weights
        self.input_size = input_size
        if w1 is None:
            self.w1 = np.zeros((input_size, hidden_size))

            for i in range(input_size):
                for j in range(hidden_size):
                    self.w1[i][j] = np.random.randn()
                    #self.w1[i][j] = 1
        elif isinstance(w1, list):
            self.w1 = np.array(w1)
        elif isinstance(w1, np.ndarray):
            self.w1 = w1

        if w2 is None:
            self.w2 = np.zeros((hidden_size, num_class))

            for i in range(hidden_size):
                for j in range(num_class):
                    self.w2[i][j] = np.random.randn()
                    #self.w2[i][j] = 1
        elif isinstance(w2, list):
            self.w2 = np.array(w2)
        elif isinstance(w2, np.ndarray):
            self.w2 = w2

        # set biases
        if b1 is None:
            self.b1 = np.ones(hidden_size)
        elif isinstance(b1, list):
            self.b1 = np.array(b1)
        else:
            self.b1 = b1

        if b2 is None:
            self.b2 = np.ones(num_class)
        elif isinstance(b2, list):
            self.b2 = np.array(b2)
        else:
            self.b2 = b2

    def one_hot(self, y):
        y = pd.Series(y)
        y = pd.get_dummies(y).values.tolist()
        return np.array(y)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)

        return a2

    def backward(self, x, y, alpha):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)

        d2 = a2-y
        d1 = np.multiply(np.dot(self.w2, d2.transpose()).transpose(), np.multiply(a1, 1-a1))

        w1_alt = x.transpose().dot(d1)
        w2_alt = a1.transpose().dot(d2)
        b1_alt = np.sum(d1, axis=0)
        b2_alt = np.sum(d2, axis=0)

        self.w1 = self.w1-(alpha*w1_alt)
        self.w2 = self.w2-(alpha*w2_alt)
        self.b1 = self.b1 - (alpha*b1_alt)
        self.b2 = self.b2 - (alpha*b2_alt)

    def loss(self, out, y):
        real = np.max(y)
        pred = np.max(out)
        return -1*real*np.log(pred)

    def accuracy(self, y, preds):
        correct = 0
        for i in range(len(y)):
            if np.max(y[i]) == np.max(preds[i]):
                correct += 1
        return correct/len(y)

    def train(self, x, y, alpha = 0.01, epoch = 10, x_max = None):
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

            for j in tqdm(range(len(x))):
                out = self.forward(x[j])
                preds.append(out)
                l.append(self.loss(out, y[j]))
                self.backward(x, y, alpha)
            
            acc.append(self.accuracy(y, preds))
            loss.append(sum(l)/len(x))
            print("epoch:", i, "accuracy:", acc[-1], "loss:", loss[-1])
            
        return acc, loss, self.w1, self.w2, self.b1, self.b2

    def predict(self, x):
        out = self.forward(x)
        return out
