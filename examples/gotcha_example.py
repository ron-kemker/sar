# -*- coding: utf-8 -*-
'''
File: gotcha_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer
from signal_processing import taylor_window

data_path ='..\..\data\GOTCHA\DATA\pass8\HH'

timer = Timer('FileIO')
with timer as _:
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=taylor_window,
                min_azimuth_angle=40,
                max_azimuth_angle=45
                )

timer = Timer("PFA")
with timer as _:
    image = PFA(sar_obj)
    
imshow(image.T)
