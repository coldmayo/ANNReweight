import numpy as np
import pandas as pd

PID2FLOAT_MAP = {0: 0, 22: 0, 211: .1, -211: .2, 321: .3, -321: .4, 130: .5, 2112: .6, -2112: .7, 2212: .8, -2212: .9, 11: 1.0, -11: 1.1, 13: 1.2, -13: 1.3}

def remap_pids(events, pid_i=3, error_on_unknown=True):
    if events.ndim == 3:
        pids = events[:,:,pid_i].astype(int).reshape((events.shape[0]*events.shape[1]))
        if error_on_unknown:
            events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP[pid] for pid in pids]).reshape(events.shape[:2])
        else:
            events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP.get(pid, 0) for pid in pids]).reshape(events.shape[:2])
    else:
        if error_on_unknown:
            for event in events:
                event[:,pid_i] = np.asarray([PID2FLOAT_MAP[pid] for pid in event[:,pid_i].astype(int)])
        else:
            for event in events:
                event[:,pid_i] = np.asarray([PID2FLOAT_MAP.get(pid, 0) for pid in event[:,pid_i].astype(int)])

def normalize(x):
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()

def preprocess_data(X):
    for x in X:
        normalize(x)
    remap_pids(X, pid_i=3)
    return X