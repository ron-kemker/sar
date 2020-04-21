# -*- coding: utf-8 -*-
'''
File: cv_example.py
Description: Example script for Civilian Vehicle Data Dome Dataset 
Author: Ronald Kemker

'''

from fileIO.cvdata import CVData
from image_formation import backProjection
from utils import imshow
from signal_processing import  hamming_window

data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
target = data_path.split('\\')[-2]

sar_obj = CVData(data_path, target,  
            polarization='vv',
            center_frequency=9.6e9, 
            bandwidth=300e6,
            taper_func=hamming_window,
            min_azimuth_angle = 41,
            max_azimuth_angle = 45,
            )

image = backProjection(sar_obj, num_x_samples=51, num_y_samples=51)
imshow(image.T)
