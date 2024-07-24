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

    df = pd.read_csv('../data/data.csv')

    x_data = df["X0"].values.reshape(-1, 1)

    preds = nn.predict(x_data)
    wei = preds[:, 0] / preds[:, 1]

    bins = np.linspace(-6, 5, 31)
    plt.hist(df["X0"], bins=bins, alpha=0.5, label='Sample MC')
    plt.hist(df["X0"], bins=bins, alpha=0.5, weights=wei, label='Weighted MC', color='k')
    plt.hist(df["X1"], bins=bins, alpha=0.5, label="Sample 'Real Data'")
    plt.legend()
    plt.savefig("../output/weighted_dist.png")
    plt.show()

if __name__ == "__main__":
    main()