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

    nn = ml.FNN(weights["input_size"], weights["hidden_size"], weights["num_class"], w1 = np.array(weights["w1"]), w2 = np.array(weights["w2"]), w3 = np.array(weights["w3"]), w4 = np.array(weights["w4"]), b1 = np.array(weights["b1"]), b2 = np.array(weights["b2"]), b3 = np.array(weights["b3"]), b4 = np.array(weights["b4"]))

    df = pd.read_csv('../data/data.csv')
    
    wei = []
    data = []
    for i in tqdm(range(len(df["X0"]))):
        out = nn.predict(np.array(df["X0"][i]))
        w = out[0][1]/out[0][0]
        data.append(w*df["X0"][i])
        wei.append(w)

    #print(wei)
    bins = np.linspace(-6, 5, 31)
    plt.hist(df["X0"], bins = bins, alpha = 0.5, label='Sample MC')
    plt.hist(df["X0"], bins=bins, alpha = 0.5, label='Weighted MC', weights = wei, color='k')
    plt.hist(df["X1"], bins = bins, alpha = 0.5, label="Sample 'Real Data'")
    plt.legend()
    plt.show()
    plt.savefig("../output/weighted_dist.png")

if __name__ == "__main__":
    main()
