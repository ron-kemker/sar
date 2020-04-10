# -*- coding: utf-8 -*-
'''

File: buildMSTARTrainTest.py
Description: Script that processes MSTAR data and breaks it up into train
             and test folds for machine learning tasks.
Author: Ronald Kemker

Download these three Files and Move into a common path (data_path)
- MSTAR Target Chips (T72 BMP2 BTR70 SLICY)
- MSTAR / IU Mixed Targets CD1 
- MSTAR / IU Mixed Targets CD2

Reference
 - [MSTAR Overview](https://www.sdms.afrl.af.mil/index.php?collection=mstar)

'''
import os
from read_mstar import readMSTARFile
from glob import glob
import numpy as np
from scipy.io import savemat

# Crops MSTAR image to 128x128 pixels
def cropMSTAR(X):
    sh = X.shape
    row= np.int((sh[0]-128) / 2)
    col = np.int((sh[1]-128) / 2)
    return X[row:row+128, col:col+128]

data_path = 'data\MSTAR\MSTAR-10_RAW'
seed = 66

# Find all valid MSTAR RAW files
files = [y for x in os.walk(data_path) for y in glob(os.path.join(x[0],'*.0*'))]

data = []
labels = []
serial_numbers = []
depression_angle = []

# Open all of the MSTAR RAW files and store them in lists
for i,f in enumerate(files):
    print('\rLoading Data: %1.1f%%' % (i/len(files)*100.0), end="")
    d, l, s, da= readMSTARFile(f)
    if d.shape[0] >= 128:
        data += [d]
        labels += [l]
        serial_numbers += [s]
        depression_angle += [da]

print("")

# Put the data into numpy arrays
X = np.zeros((len(data), 128, 128, 2), dtype=np.float32)
y = np.zeros((len(data),), np.int32)
depression_angle = np.int32(depression_angle)
unique_labels = list(set(labels))
for i,d in enumerate(data):       
    print('\rBuilding Dataset: %1.1f%%' % (i/len(data)*100.0), end="")
    if d.shape[0] > 128 or d.shape[1] > 128:
        d = cropMSTAR(d)
    X[i] = d
    y[i] = unique_labels.index(labels[i])

# Training Data is for depression_angle=17
train_idx = depression_angle == 17
X_train = X[train_idx]
y_train = y[train_idx]
depression_angle_train = depression_angle[train_idx]

# Testing data is for depression_angle=15,30, and 45
test_idx = depression_angle !=  17
X_test = X[test_idx]
y_test = y[test_idx]
depression_angle_test = depression_angle[test_idx]

# Store them in a .mat file for future work
save_dict = {'X_train' : X_train, 'y_train': y_train, 
              'depression_angle_train' : depression_angle_train, 
              'X_test' : X_test , 'y_test' : y_test,
              'depression_angle_test' : depression_angle_test,
              'class_labels' : unique_labels}

savemat('mstar10.mat', save_dict)
