# -*- coding: utf-8 -*-
'''
File: meanshit_tf_example.py
Description: This is an example script to run my Tensorflow accelerated 
             version of the Mean-Shift clustering algorithm on MSTAR-10.
Author: Ronald Kemker

'''

import numpy as np
from machine_learning.mean_shift_tf import MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

phase_channel = True

# Load MSTAR-10
mat = loadmat('../../mstar10.mat')
X_train = mat['X_train']
X_test = mat['X_test']
y_train = mat['y_train'][0]
y_test = mat['y_test'][0]
da_test = mat['depression_angle_test'][0]
del mat

# Keep the phase information or discard it
if phase_channel == False:
    X_train = X_train[:,:,:,0]
    X_test = X_test[:,:,:,0]
    num_channels = 1
else:
    num_channels = 2

# Size of Datasets
train_shape = X_train.shape
test_shape = X_test.shape

# Reshape to N x F
X_train = X_train.reshape(train_shape[0], -1)
X_test = X_test.reshape(test_shape[0], -1)

# Scale Feature-wise [0,1]
sclr = MinMaxScaler()
X_train = sclr.fit_transform(X_train)
X_test = sclr.transform(X_test)

# Reduce to 3-dimensions for plot reasons
pca = PCA(3)
X_train =pca.fit_transform(X_train)
X_test =pca.transform(X_test)

# Perform Mean-Shift Clustering 
ms = MeanShift(bandwidth=2.5, min_bin_freq=10)
ms.fit(X_train)
tx = ms.predict(X_train)
ty = ms.predict(X_test)
n_clusters = np.max(tx)+1

# Display Graphics
fig = plt.figure(1, figsize=[10,10])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2] , c=tx, alpha=0.9, 
           marker='o', cmap='tab20')
plt.title('Mean-Shift CLustering of MSTAR-10 Train (%d Clusters)'%n_clusters)

for i, d in enumerate([15, 30, 45]):
    idx = da_test==d
    fig = plt.figure(2+i, figsize=[10,10])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test[idx,0], X_test[idx,1], X_test[idx,2] , c=ty[idx], 
               alpha=0.9, marker='o', cmap='tab20')
    msg = 'Mean-Shift Clustering of MSTAR-10 Test (Graze=%d deg)' 
    plt.title(msg % d) 
    