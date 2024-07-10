import uproot
import numpy as np
import pandas
import model as ml
from sklearn.model_selection import train_test_split

columns = ['hSPD', 'pt_b', 'pt_phi', 'vchi2_b', 'mu_pt_sum']

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

nn = ml.FNN(2, 5, 2, epoch = 15)
nn.train(x_train, y_train)
