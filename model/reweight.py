import numpy as np
import json
import model as ml
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

def main():

    with open("../saved_models/FNNweights.json") as f:
        weights = json.loads(f.read())

    nn = ml.FNN(weights["input_size"], weights["hidden_size"], weights["num_class"], w1 = np.array(weights["w1"]), w2 = np.array(weights["w2"]), b1 = np.array(weights["b1"]), b2 = np.array(weights["b2"]))

    df = pd.read_csv('../data/data.csv')
    
    wei = []
    data = []
    for i in tqdm(range(len(df["X0"]))):
        out = nn.predict(np.array(df["X0"][i]))
        w = out[0][0]/out[0][1]
        data.append(w*df["X0"][i])
        wei.append(w)

    #print(wei)
    bins = np.linspace(-6, 5, 31)
    plt.hist(df["X0"], bins = bins, alpha = 0.5, label='0')
    plt.hist(data, bins=bins, alpha = 0.5, label='1', color='k')
    plt.hist(df["X1"], bins = bins, alpha = 0.5, label='2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
