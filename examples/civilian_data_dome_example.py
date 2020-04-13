# -*- coding: utf-8 -*-
'''
File: civilian_data_dome_example.py
Description: Loads a Civilian Data Dome datafile, forms an image, and 
             displays the resulting image. 
Author: Ronald Kemker

'''

from fileIO.cvdata import CVData
from fileIO.gotcha import GOTCHA
from image_formation import backProjection, image_projection
from utils import imshow
from signal_processing import taylor_window, hamming_window

mode = 'gotcha'

if mode == 'cv':

    data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
    target = data_path.split('\\')[-2]

    data = CVData(data_path, target,  
                polarization='vv',
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=hamming_window,
                )
    Nx=41
    Ny=41
    Wx=10
    Wy=10
    x0=0
    y0=0
    
elif mode == 'gotcha':
    data_path ='..\..\data\GOTCHA\DATA\pass8\HH\data_3dsar_pass8_az001_HH.mat'
    data = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=taylor_window,
                )
    Nx, Ny, Wx, Wy, x0, y0 = data.get_image_plane_projection_params()

else:
    raise ValueError('Only supports "cv" and "gotcha".')

image_plane = image_projection(data, Nx, Ny, Wx, Wy, x0, y0)
image = backProjection(data, image_plane, fft_samples=512)
imshow(image, image_plane['x_vec'], image_plane['y_vec'])
