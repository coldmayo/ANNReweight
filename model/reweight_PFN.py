import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PFN as ml
from utils import *

def main():

    with open("../saved_models/PFNweights.json") as f:
        weights = json.loads(f.read())

    f_wei = weights["f_weights"]
    f_bias = weights["f_bias"]
    phi_wei = weights["Phi_weights"]
    phi_bias = weights["Phi_bias"]

    print("Start data processing...")
    test_dataset_0 = np.load('../data/test1D_default.npz')
    test_dataset_1 = np.load('../data/test_3D_known.npz')

    indices = np.random.choice(test_dataset_0['jet'].shape[0], 10**5, replace=False)

    X0_test = preprocess_data(test_dataset_0['jet'][indices])
    X1_test = preprocess_data(test_dataset_1['jet'][indices])
    print("Done!")
    nn = ml.PFN(phi_wei=phi_wei, phi_bias=phi_bias, f_wei=f_wei, f_bias=f_bias)
    preds = nn.predict(X1_test)
    wei = preds[:, 0]/preds[:, 1]
    print(wei)

    plt.figure(figsize=(6,5))
    bins = np.linspace(0,40,21)
    plt.hist(test_dataset_0['multiplicity'][indices], bins = bins, alpha = 0.5, label="0")
    plt.hist(test_dataset_1['multiplicity'][indices], bins = bins, weights=wei, histtype='step', color='k', label="1->0")
    plt.hist(test_dataset_1['multiplicity'][indices], bins = bins, alpha = 0.5, label="1")
    #plt.hist(test_dataset_0['number_of_kaons'][indices], bins = bins, alpha = 0.5, label="0")
    #plt.hist(test_dataset_1['number_of_kaons'][indices], bins = bins, weights=wei, histtype='step', color='k', label="1->0")
    #plt.hist(test_dataset_1['number_of_kaons'][indices], bins = bins, alpha = 0.5, label="1")
    plt.legend()
    plt.savefig("../output/weighted_dist_PFN.png")
    plt.show()


if __name__ == "__main__":
    main()