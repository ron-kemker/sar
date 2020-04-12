# -*- coding: utf-8 -*-
'''
File: timer_example.py
Description: Demonstrates timer function while testing the multiprocessing
             functionality of the backprojection algorithm.
Author: Ronald Kemker

'''

from fileIO.cvdata import CVData
from image_formation import backProjection
from utils import Timer

data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
num_exp = 10

target = data_path.split('\\')[-2]


cvdata = CVData(data_path, target,  
            polarization='vv',
            center_frequency=9.6e9, 
            bandwidth=300e6,
            taper_flag=True,
            verbose=False,
            )

timer = Timer()

with timer as t:
    for i in range(num_exp):
        image = backProjection(cvdata)
    
print('The average time is %1.2f seconds' % (timer.delta/num_exp))