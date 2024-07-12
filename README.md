# Reweighting using Neural Networks

In this project, I will be reweighting Geant4 histograms using Machine Learning. For more information check out <a href="https://arogozhnikov.github.io/hep_ml/reweight.html">this</a> web page.

## Directory breakdown
- data/
	- As the name suggests, this is where data is stored. From root to csv files
- model/
	- Where all of the code relating to building, training, and testing of the models takes place
- saved_models/
	- This directory is created once the user trains their first model. Information about the trained model is stored here (weights, bias, etc)
- useful_papers/
	- Papers referenced for this project (helpful for writing references for a paper)
- output/
	- Shows the output of the model (loss function, reweighted plots, etc)

## Important Links
Data:
- <a href="https://zenodo.org/records/3518708#.XbN4MJNKJOQ">Zenodo</a>

## Running the code
### Linux
1. Create Virtual Envirement
```bash
$ python -m venv .venv
```
2. Install packages
```bash
$ pip install -r requirements.txt
```
3. Use run.sh for running files
```bash
$ cd model
$ ./run.sh train.py 10
```

### Windows
1. Create Virtual Envirement
```bash
$ python -m venv .venv
```
2. Install packages
```bash
$ pip install -r requirements.txt
```
3. Run files
```bash
$ .venv\Scripts\activate.bat
$ cd model
$ python -u train.py 10
```
