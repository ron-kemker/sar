# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 07:52:48 2020

Title: MSTAR-10_CNN_Example.py
Purpose: Test out contrastive embedding loss effect on SAR ATR

@author: Ronald Kemker
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam
from keras.metrics import categorical_accuracy as ca
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import categorical_crossentropy
from sklearn.preprocessing import StandardScaler
from mstar_cnn_library import MSTAR10Net as CNN
import tensorflow as tf
import random

seed = 66 # Improve reproducability witha random seed
mb = 32 # Mini-Batch Size
l2_scale = 1e-4 # Weight-decay hyperparamter (used for L2-regularization)
num_epochs = 200 # Number of training epochs (arbitrarily selected for demo)
save_file = 'mstar10.h5'
kernel_initializer = 'he_normal' # Kernel initialization scheme
activation = 'elu'
phase_channel = False # Include phase data or not?
perc_train = 0.9 # Percentage reserved for training in train/val folds
stest_translation = 0 # Random spatial translation for test data
lmda = 0.01 # Weight of the constrative embedding loss

# I suspected that people were cheating, i.e., using testing data in place of
# validation data...this was a test to validate that suspicion. 
cheat = False 

"""
This function just helps balance the training set to have an equal number of
"same-class" and "different-class" image pairs.
"""
def FindSimDiffPair(y):
    y = np.int32(np.argmax(y, 1))
    sh = y.shape

    idx_list = []
    for i in range(np.max(y)+1):
        idx_list += [np.argwhere(y == i)[:,0]]
        
    dissim_idx = np.arange(np.max(y)+1)
    pair_idx = -1 * np.ones(sh, np.int32)
    for i in range(sh[0]): # Dissimilar
        if i % 2:
            tmp = random.choice(dissim_idx[y[i]!=dissim_idx])
            pair_idx[i] = random.choice(idx_list[tmp])
        else:
            pair_idx[i] = random.choice(idx_list[y[i]])
            
    return pair_idx
        
"""
This is the data generator for the fit function.
"""
def MyGenerator(X_train, y_train, batch_size=32, augment=True):
    
    sh =X_train.shape
    epoch_idx = np.arange(sh[0])
    mb = int(batch_size/2)
    flip_idx = np.arange(sh[0])
    while 1:
        
        np.random.shuffle(epoch_idx)
        pair_idx = FindSimDiffPair(y_train[epoch_idx])
        if augment == True:
            np.random.shuffle(flip_idx)
            to_flip = flip_idx[:int(sh[0]/2)]
            X_train[to_flip] = X_train[to_flip, :, ::-1]
                        
        for i in range(0, sh[0]-mb, mb):
            iter_idx = np.append(epoch_idx[i:i+mb] , pair_idx[i:i+mb])
            yield X_train[iter_idx], y_train[iter_idx]

"""
This is the sum of the classification and contrastive losses
"""
def custom_loss(k_layer , lmda=0.1):
    def cust_loss(y_true, y_pred):
        
        slice_idx = int(mb/2)
        delta = k_layer[:slice_idx] - k_layer[slice_idx:]
        norm = tf.reduce_sum(tf.square(delta) , axis=-1)
        
        truth1 = tf.argmax(y_true[:slice_idx], -1)
        truth2 = tf.argmax(y_true[slice_idx:], -1)
        
        one = tf.constant(1, tf.float32)
        same = tf.cast(tf.equal(truth1,truth2), tf.float32)
        
        L1 =tf.multiply( tf.nn.relu(norm) , same)
        L2 = tf.multiply( tf.nn.relu(one-norm) , one-same )
        L = 0.5*(L1 + L2) * lmda 
        L = tf.concat([L , L],0)

        soft_act = tf.nn.softmax(y_pred)
        softmax_loss = categorical_crossentropy(y_true, soft_act)
        
        return L + softmax_loss
    return cust_loss

# Load MSTAR-10
mat = loadmat('../../mstar10.mat')
X_train = mat['X_train']
X_test = mat['X_test']
y_train = mat['y_train'][0]
y_test = mat['y_test'][0]
da_test = mat['depression_angle_test'][0]
class_labels = mat['class_labels']
del mat

# Keep the phase information or discard it
if phase_channel == False:
    X_train = X_train[:,:,:,0:1]
    X_test = X_test[:,:,:,0:1]
    num_channels = 1
else:
    num_channels = 2

# Feature-Wise Scale to zero-mean/unit variance
train_shape = X_train.shape
test_shape = X_test.shape

X_train = X_train.reshape(train_shape[0], -1)
X_test = X_test.reshape(test_shape[0], -1)

sclr = StandardScaler()
X_train = sclr.fit_transform(X_train).reshape(train_shape)
X_test = sclr.transform(X_test).reshape(test_shape)

# Build Train/Validation Folds
if cheat:
    idx = da_test == 15
    X_val = np.copy(X_test[idx])
    y_val = np.copy(y_test[idx])
else:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                       train_size=perc_train,
                                                        random_state=seed,
                                                        stratify=y_train)

# One-Hot Vector Representation of Truth Labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Define the model       
model = CNN(num_channels=num_channels, l2_scale=l2_scale,
            activation=activation)

k_layer = model.get_layer('k_layer').output

# Compile Keras model w/ Adam optimizer
opt = Nadam() 

# Define the loss
model.compile(opt, loss=custom_loss(k_layer, lmda), metrics=[ca])

# Define Callbacks
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,verbose=1)
stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1,
                      restore_best_weights=True)
ckpt = ModelCheckpoint(save_file, monitor='val_loss', 
                        save_best_only=True, verbose=1)

# Define a custom generator for the training and validation data
train_gen = MyGenerator
val_gen = MyGenerator

# Fit the CNN with 
log = model.fit_generator(train_gen(X_train, y_train, batch_size=mb), 
                    steps_per_epoch= int(2*X_train.shape[0] / mb), 
                    epochs=num_epochs, 
                    validation_data=val_gen(X_val,y_val, batch_size=mb,
                                            augment=False), 
                    validation_steps=int(2*X_val.shape[0] / (mb)),
                    shuffle=True, callbacks=[lr,ckpt, stop])

# Run test data through trained CNN
y_pred = model.predict(X_test) # One-hot encoded
y_pred = np.argmax(y_pred, 1) # Convert to integer predictions

# Display Loss/Accuracy Curves
history = log.history
plt.figure(1, figsize = (10,7))
plt.subplot(2,1,1)
plt.title('Classify MSTAR-10 Targets')
x = np.arange(len(history['loss'])) + 1
plt.plot(x, history['loss'], 'r-', x, history['val_loss'], 'b-')
plt.xlabel('Epochs')
plt.ylabel('Crossentropy Loss')
plt.legend(['Train', 'Validation'])
plt.subplot(2,1,2)
plt.plot(x, 100.0*np.array(history['categorical_accuracy']), 'r-', 
          x, 100.0*np.array(history['val_categorical_accuracy']), 'b-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy [%]')
plt.legend(['Train', 'Validation'])

# Plot normalized confusion matrix
idx = da_test == 15
N = np.max(y_test[idx]) + 1
cm = confusion_matrix(y_test[idx], y_pred[idx], normalize='true')
df_cm = pd.DataFrame(cm, range(N), range(N))
plt.figure(2, figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix for MSTAR-10 (15 Degree Depression Angle)')
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')

# Assess Depression Angle Impact
for d in [15, 30, 45]:
    
    print('\nWhen the depression angle is %d degrees...' % d)
    idx = da_test == d
    overall_accuracy = np.mean(y_test[idx]==y_pred[idx]) * 100.0
    mean_class_accuracy = balanced_accuracy_score(y_test[idx], 
                                                  y_pred[idx]) * 100.0
    kappa_statistic = cohen_kappa_score(y_test[idx], y_pred[idx])
    print("Overall Accuracy is %1.1f%%" % overall_accuracy)
    print("Mean-Class Accuracy is %1.1f%%" % mean_class_accuracy)
    print("Cohen's Kappa Statistic is %1.3f" % kappa_statistic)

# Performance across the entire test set
print('\nOverall performance...')
overall_accuracy = accuracy_score(y_test, y_pred) * 100.0
mean_class_accuracy = balanced_accuracy_score(y_test, y_pred) * 100.0
kappa_statistic = cohen_kappa_score(y_test, y_pred)
print("Overall Accuracy is %1.1f%%" % overall_accuracy)
print("Mean-Class Accuracy is %1.1f%%" % mean_class_accuracy)
print("Cohen's Kappa Statistic is %1.3f" % kappa_statistic)