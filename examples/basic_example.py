# -*- coding: utf-8 -*-
'''
File: basic_example.py
Description: Example script for Civilian Vehicle and GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.cvdata import CVData
from fileIO.gotcha import GOTCHA
from image_formation import backProjection, image_projection
from image_formation import polar_format_algorithm as PFA
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
    data_path ='..\..\data\GOTCHA\DATA\pass8\HH'
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=taylor_window,
                min_azimuth_angle=40,
                max_azimuth_angle=45
                )

    Nx = sar_obj.range_pixels
    Ny = sar_obj.cross_range_pixels

else:
    raise ValueError('Only supports "cv" and "gotcha".')

# image_plane = image_projection(sar_obj, Nx, Ny, Wx, Wy)
image = PFA(sar_obj)
imshow(image.T)
