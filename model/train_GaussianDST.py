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
    weights = {"w1":[], "w2":[], "b1":[], "b2":[], "input_size": 0, "hidden_size": 0, "num_class": 0}

    # root data

    #with uproot.open("../data/MC_distribution.root") as og_file:
        #original_tree = og_file['tree']
        #original = original_tree.arrays(library='pd')

    #print(original)

    #with uproot.open("../data/RD_distribution.root") as og_file:
        #target_tree = og_file['tree']
        #target = target_tree.arrays(library='pd')

    #original['index'] = range(0, len(original))
    #target['index'] = range(0, len(target))

    #print(original)

    #x = []
    #y = []

    #print("Start data processing")

    #for i in range(len(original['index'])):
        #x.append([original['hSPD'][i], original['index'][i]])
        #y.append([0, 1])

    #for i in range(len(target['index'])):
        #x.append([target['hSPD'][i], target['index'][i]])
        #y.append([1, 0])

    #print("Done")

    #df = pd.DataFrame(data={"MC":original['hSPD'], "MC_index":original["index"], 'RD':target['hSPD'], 'RD_index':target["index"],})
    print("Start Data collecting...")
    mu0 = 0
    mu1 = 1
    var0 = 1
    var1 = 1.3
    X0 = np.random.normal(mu0, var0, 5000)
    X1 = np.random.normal(mu1, var1, 5000)
    plt.hist(X0)
    plt.hist(X1)
    plt.show()
    Y0 = np.zeros(5000)
    Y1 = np.ones(5000)
    x = np.concatenate((X0, X1))
    y = np.concatenate((Y0, Y1))
    df = pd.DataFrame(data = {'X0':X0, 'Y0':Y0, 'X1':X1, 'Y1':Y1})
    df.to_csv("../data/data.csv")
    print(x.shape)
    print("Got it!")
    #df.to_csv("../data/data.csv")
    x = x.reshape((5000*2, 1))
    nn = ml.FNN(1, 5, 2)
    acc, loss, weight1, weight2, bias1, bias2 = nn.train(x, y, epoch=epochs)

    weights["w1"] = weight1.tolist()
    weights["w2"] = weight2.tolist()
    weights["b1"] = bias1.tolist()
    weights["b2"] = bias2.tolist()
    weights["input_size"] = 1
    weights["hidden_size"] = 5
    weights["num_class"] = 2

    with open("../saved_models/FNNweights.json", 'w') as f:
        json.dump(weights, f)

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
