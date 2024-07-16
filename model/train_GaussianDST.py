import uproot
import numpy as np
import pandas as pd
import model as ml
import json
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main(argv, argc):

    if argc == 2:
        epochs = int(argv[1])
    else:
        epochs = 3
        
    #columns = ['hSPD', 'pt_b', 'pt_phi', 'vchi2_b', 'mu_pt_sum']
    weights = {"w1":[], "w2":[], "w3":[], "w4":[], "b1":[], "b2":[], "b3":[], "b4":[], "input_size": 0, "hidden_size": 0, "num_class": 0}

    print("Start Data collecting...")
    mu0 = 0
    mu1 = 1
    var0 = 1
    var1 = 1.3
    leng = 100000
    X0 = np.random.normal(mu0, var0, int(leng/2))
    X1 = np.random.normal(mu1, var1, int(leng/2))

    #plt.hist(X0)
    #plt.hist(X1)
    #plt.show()

    Y0 = np.zeros(int(leng/2))
    Y1 = np.ones(int(leng/2))
    x = np.concatenate((X0, X1))
    y = np.concatenate((Y0, Y1))
    df = pd.DataFrame(data = {'X0':X0, 'Y0':Y0, 'X1':X1, 'Y1':Y1})
    df.to_csv("../data/data.csv")
    print(x.shape)
    print("Got it!")
    #df.to_csv("../data/data.csv")
    x = x.reshape((leng, 1))
    nn = ml.FNN(1, 20, 2)
    acc, loss, weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4 = nn.train(x, y, epoch=epochs)

    # create loss plots:
    epochs = np.arange(len(loss))
    plt.plot(epochs, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Curve")
    plt.savefig("../output/loss.png")

    weights["w1"] = weight1.tolist()
    weights["w2"] = weight2.tolist()
    weights["w3"] = weight3.tolist()
    weights["w4"] = weight4.tolist()
    weights["b1"] = bias1.tolist()
    weights["b2"] = bias2.tolist()
    weights["b3"] = bias3.tolist()
    weights["b4"] = bias4.tolist()
    weights["input_size"] = 1
    weights["hidden_size"] = 20
    weights["num_class"] = 2

    with open("../saved_models/FNNweights.json", 'w') as f:
        json.dump(weights, f)

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
