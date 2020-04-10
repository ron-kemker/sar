# -*- coding: utf-8 -*-
'''

File: process_all_cv_data.py
Description: Finds all of the CV .mat files in a directory and loads them
Author: Ronald Kemker

'''

import numpy as np
from cvdata import CVData
from glob import glob
import os

data_path = '..\data\Civilian Vehicles\Domes'
num_x_samples = 41
num_y_samples = 41
polarization='vv'
min_azimuth_angle = 0
max_azimuth_angle=360 
bandwidth=300e6 
center_frequency=9.6e9
scene_extent_x = 10 
scene_extent_y=10
fft_samples=512
                
files = [y for x in os.walk(data_path) for y in glob(os.path.join(x[0],'*.mat'))]

data = np.zeros((len(files),num_y_samples,num_x_samples,2), np.float32)
truth = []
elevation = np.zeros((len(files), ), np.float32)

for i, f in enumerate(files):
    
    print('Image %d/%d:' % (i+1,len(files)), end="")
    
    target = f.split('\\')[-2]
    
    cvdata = CVData(f, target, polarization=polarization,
                 min_azimuth_angle = min_azimuth_angle, 
                 max_azimuth_angle=max_azimuth_angle, 
                 bandwidth=bandwidth, 
                 center_frequency=center_frequency,
                 taper_flag = True, 
                 scene_extent_x = scene_extent_x, 
                 scene_extent_y=scene_extent_y, 
                 fft_samples=fft_samples, 
                 num_x_samples=num_x_samples,
                 num_y_samples=num_y_samples, 
                 scene_center_x = 0, 
                 scene_center_y=0,
                 verbose = False, 
                 n_jobs=1,
                 single_precision=True)

    data[i] = cvdata.getAmpPhaseChannels()
    truth += [target]
    elevation[i] = cvdata.elevation