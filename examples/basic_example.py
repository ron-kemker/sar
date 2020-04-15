# -*- coding: utf-8 -*-
'''
File: basic_example.py
Description: Example script for Civilian Vehicle and GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.cvdata import CVData
from fileIO.gotcha import GOTCHA
from image_formation import backProjection, image_projection
from utils import imshow
from signal_processing import taylor_window, hamming_window

mode = 'gotcha' # other option is 'cv'

if mode == 'cv':

    data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
    target = data_path.split('\\')[-2]

    sar_obj = CVData(data_path, target,  
                polarization='vv',
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=hamming_window,
                )

    Nx=51
    Ny=51

elif mode == 'gotcha':
    data_path ='..\..\data\GOTCHA\DATA\pass8\HH\data_3dsar_pass8_az001_HH.mat'
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=taylor_window,
                )
    
    Nx=201
    Ny=201

else:
    raise ValueError('Only supports "cv" and "gotcha".')

Wx=int(sar_obj.Wx)
Wy=int(sar_obj.Wy)
image_plane = image_projection(sar_obj, Nx, Ny, Wx, Wy)
image = backProjection(sar_obj, image_plane, fft_samples=512)
imshow(image, image_plane['x_vec'], image_plane['y_vec'])
