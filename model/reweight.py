import numpy as np
import json
import model as ml
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

with open("../saved_models/FNNweights.json") as f:
    weights = json.loads(f.read())

nn = ml.FNN(weights["input_size"], weights["hidden_size"], weights["num_class"], w1 = np.array(weights["w1"]), w2 = np.array(weights["w2"]))

df = pd.read_csv('../data/data.csv')

newMC = []
wei = []
for i in tqdm(range(len(df["MC"]))):
    out = nn.predict(np.array([df["MC"][i], df["MC_index"][i]]))
    w = out[0]/out[1]
    wei.append(w)
    newMC.append(df["MC"][i]*w)

print(newMC)
plt.plot(newMC)
plt.plot(df["RD"])
plt.show()