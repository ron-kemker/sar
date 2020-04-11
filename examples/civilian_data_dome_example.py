# -*- coding: utf-8 -*-
'''
File: civilian_data_dome_example.py
Description: Loads a Civilian Data Dome datafile, forms an image, and 
             displays the resulting image. 
Author: Ronald Kemker

'''

from cvdata import CVData
from image_formation import backProjection
from utils import imshow

data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
target = data_path.split('\\')[-2]

cvdata = CVData(data_path, target,  
            polarization='vv',
            center_frequency=9.6e9, 
            bandwidth=300e6,
            taper_flag=True,
            )

image = backProjection(cvdata)
imshow(image, cvdata.x_vec, cvdata.y_vec)