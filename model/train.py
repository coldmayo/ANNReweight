import uproot
import numpy as np
import pandas as pd
import model as ml
import json
import sys
from sklearn.model_selection import train_test_split

def main(argv, argc):

    if argc == 2:
        epochs = int(argv[1])
        
    columns = ['hSPD', 'pt_b', 'pt_phi', 'vchi2_b', 'mu_pt_sum']
    weights = {"w1":[], "w2":[], "b1":[], "b2":[], "input_size": 0, "hidden_size": 0, "num_class": 0}

    with uproot.open("../data/MC_distribution.root") as og_file:
        original_tree = og_file['tree']
        original = original_tree.arrays(library='pd')

    #print(original)

    with uproot.open("../data/RD_distribution.root") as og_file:
        target_tree = og_file['tree']
        target = target_tree.arrays(library='pd')

    original['index'] = range(0, len(original))
    target['index'] = range(0, len(target))

    #print(original)

    x = []
    y = []

    print("Start data processing")

    for i in range(len(original['index'])):
        x.append([original['hSPD'][i], original['index'][i]])
        y.append([0, 1])

    for i in range(len(target['index'])):
        x.append([target['hSPD'][i], target['index'][i]])
        y.append([1, 0])

    print("Done")

    df = pd.DataFrame(data={"MC":original['hSPD'], "MC_index":original["index"], 'RD':target['hSPD'], 'RD_index':target["index"],})
    df.to_csv("../data/data.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

    nn = ml.FNN(2, 5, len(y[0]))
    acc, loss, weight1, weight2, bias1, bias2 = nn.train(x_train, y_train, epoch=epochs)

    weights["w1"] = weight1.tolist()
    weights["w2"] = weight2.tolist()
    weights["b1"] = bias1.tolist()
    weights["b2"] = bias2.tolist()
    weights["input_size"] = 2
    weights["hidden_size"] = 5
    weights["num_class"] = len(y[0])

    with open("../saved_models/FNNweights.json", 'w') as f:
        json.dump(weights, f)

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
