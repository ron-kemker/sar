# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 07:52:48 2020

Title: MSTAR_CNN_Library.py
Purpose: The CNN used for the MSTAR-10 Example

@author: Ronald Kemker
"""

from keras.models import Model
from keras.layers import Input, Activation, BatchNormalization
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras.regularizers import l2

def MSTAR10Net(num_channels=1, activation='elu', l2_scale=5e-4):
    """The CNN architecture used for this experiment.
    
    @author: Ronald Kemker
    
    This code instantiates the CNN model with the default arguments.
    model = MSTAR10Net()

    # Arguments
        num_channels: integer >= 1.  Number of input channels.
        activation: string.  The Keras activation function.
        l2_scale: float >= 0.  The weight decay term.
    
    # References
        - [Successive Embedding and Classification Loss for Aerial Image 
          Classification](https://arxiv.org/pdf/1712.01511.pdf)
    """
    
    input_t = Input((128, 128, num_channels), name='input')
    
    # 128 x 128
    x = Conv2D(16, 5, padding='valid', 
               kernel_regularizer=l2(l2_scale))(input_t)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)
    
    # 62 x 62
    x = Conv2D(32, 3, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(32, 3, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)
    
    # 29 x 29
    x = Conv2D(64, 3, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 4, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)
    
    # 12 x 12
    x = Conv2D(128, 3, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # 8 x 8
    x = Conv2D(128, 3, padding='valid', 
               kernel_regularizer=l2(l2_scale))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)
    
    # 3 x 3
    x = Flatten()(x) 
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    k_layer = Dense(10, name='k_layer')(x)
    
    model = Model(inputs=input_t, outputs=k_layer)

    return model