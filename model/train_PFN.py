cupy = False
try:
    import cupy as np
    cupy = True
except ImportError:
    import numpy as np
    cupy = False  

from sklearn.model_selection import train_test_split
from utils import *
import PFN as ml
import sys
import matplotlib.pyplot as plt
import json

def main(argv, argc):

    weights = {"f_weights": [], "f_bias": [], "Phi_weights": [], "Phi_bias": []}

    if argc == 2:
        epochs = int(argv[1])
    else:
        epochs = 3

    print("Load Data, this might take a while...")
    data = np.load("../data/3D_train.npz", mmap_mode='r')
    X = data['X']
    Y = data['Y']

    X = preprocess_data(X)
    print("Data Loaded")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
    if cupy == True:
        import cupy as cp
        X_train = cp.asarray(X_train)
        Y_train = cp.asarray(Y_train)

    print(X_train.shape, type(X_train))
    print(Y_train.shape)

    print(X_val.shape)
    print(Y_val.shape)

    nn = ml.PFN()
    acc, loss, f_wei, f_bias, phi_wei, phi_bias = nn.train(X_train, Y_train, epoch=epochs, batch_size=1000, x_max=10**5)

    epochs = np.arange(len(loss))
    if cupy:
        import cupy as cp
        loss = cp.asnumpy(cp.array(loss).get()).tolist()
        plt.plot(epochs, loss)
    else:
        plt.plot(epochs, loss)
    
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Curve")
    plt.savefig("../output/loss_PFN.png")
    
    weights["f_bias"] = [f_bias[0].tolist(), f_bias[1].tolist(), f_bias[2].tolist()]
    weights["f_weights"] = [f_wei[0].tolist(), f_wei[1].tolist(), f_wei[2].tolist()]
    weights["Phi_bias"] = [phi_bias[0].tolist(), phi_bias[1].tolist(), phi_bias[2].tolist()]
    weights["Phi_weights"] = [phi_wei[0].tolist(), phi_wei[1].tolist(), phi_wei[2].tolist()]

    with open("../saved_models/PFNweights.json", 'w') as f:
        json.dump(weights, f)

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))