import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model as ml

def main():
    with open("../saved_models/FNNweights.json") as f:
        weights = json.loads(f.read())

    w1 = np.array(weights["w1"])
    w2 = np.array(weights["w2"])
    w3 = np.array(weights["w3"])
    w4 = np.array(weights["w4"])
    b1 = np.array(weights["b1"])
    b2 = np.array(weights["b2"])
    b3 = np.array(weights["b3"])
    b4 = np.array(weights["b4"])

    nn = ml.FNN(
        input_size=weights["input_size"],
        hidden_size=weights["hidden_size"],
        num_class=weights["num_class"],
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        b1=b1,
        b2=b2,
        b3=b3,
        b4=b4
    )

    #df = pd.read_csv('../data/data.csv')

    #x_data = df["X0"].values.reshape(-1, 1)
    mu0 = 0
    mu1 = 1
    var0 = 1
    var1 = 1.3
    X0_val = np.random.normal(mu0, var0, 10**5)
    X1_val = np.random.normal(mu1, var1, 10**5)

    preds = nn.predict(X0_val.reshape((10**5, 1)))
    wei = (preds[:, 1]) / (preds[:, 0])
    print(preds)
    print(wei)

    bins = np.linspace(-6,5,31)
    plt.hist(X0_val, bins = bins, alpha = 0.5, label = r'$\mu=0$')
    plt.hist(X0_val, bins = bins, label = r'$\mu=0$ weighted', weights=wei, histtype='step', color='k')
    plt.hist(X1_val, bins = bins, alpha = 0.5, label = r'$\mu=1$')
    plt.legend()
    plt.savefig("../output/weighted_dist.png")
    plt.show()

if __name__ == "__main__":
    main()