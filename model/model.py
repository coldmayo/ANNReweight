import numpy as np
from tqdm import tqdm

class FNN:
    def __init__(self, input_size, hidden_size, num_class):
        self.w1 = np.zeros((input_size, hidden_size))
        self.w2 = np.zeros((hidden_size, num_class))

        for i in range(input_size):
            for j in range(hidden_size):
                self.w1[i][j] = np.random.randn()

        for i in range(hidden_size):
            for j in range(num_class):
                self.w2[i][j] = np.random.randn()

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, x):
        z1 = x.dot(self.w1)
        a1 = self.sigmoid(z1)

        z2 = a1.dot(self.w2)
        a2 = self.sigmoid(z2)

        return a2

    def backward(self, x, y, alpha):
        z1 = x.dot(self.w1)
        a1 = self.sigmoid(z1)

        z2 = a1.dot(self.w2)
        a2 = self.sigmoid(z2)

        d2 = a2-y
        d1 = np.multiply(self.w2.dot(d2.transpose()).transpose(), np.multiply(a1, 1-a1))

        w1_alt = x.transpose().dot(d1)
        w2_alt = a1.transpose().dot(d2)

        self.w1 = self.w1-(alpha*(w1_alt))
        self.w2 = self.w2-(alpha*(w2_alt))

    def loss(self, out, y):
        real = np.max(y)
        pred = np.max(out)
        return -real*np.log(pred)

    def accuracy(self, y, preds):
        correct = 0
        for i in range(len(y)):
            if np.max(y[i]) == np.max(preds[i]):
                correct += 1
        return correct/len(y)

    def train(self, x, y, alpha = 0.01, epoch = 10, x_max = 10000):
        acc = []
        loss = []
        x = np.array(x)
        y = np.array(y)
        print("Start Training: ")
        for i in range(epoch):
            l = []
            preds = []
            if len(x) > x_max:
                x_old = x
                y_old = y

                x = []
                y = []

                for k in np.random.choice(len(x_old), x_max):
                    x.append(x_old[k])
                    y.append(y_old[k])

                x = np.array(x)
                y = np.array(y)

            for j in tqdm(range(len(x))):
                out = self.forward(x[j])
                preds.append(out)
                l.append(self.loss(out, y[j]))
                self.backward(x, y, alpha)
            
            acc.append(self.accuracy(y, preds))
            loss.append(sum(l)/len(x))
            print("epoch:", i, "accuracy:", acc[-1], "loss:", loss[-1])
            
        return acc, loss, self.w1, self.w2

    def predict(self):
        out = self.forward(x)
        return out
