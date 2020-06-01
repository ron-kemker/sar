# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:42:04 2020

@author: Master
"""

import numpy as np
from fileIO.lsi import LSIReader, LSIWriter

# Int32 Header Values
pixels_per_column=100
pixels_per_row=100
channels=4
numeric_type_indicator=2
apodization_type=0
remap_type=0
image_plane_indicator=1
    
# Float32 Header Values
rf_center_frequency        = 9.6e9
rf_bandwidth               = 640e6
dwell_angle                = 5
cone_angle                 = 90
graze_angle                = 45
twist_angle                = 0
column_sample_spacing      = 0.25 
row_sample_spacing         = 0.25
column_oversampling_factor = 1
row_oversampling_factor    = 1
column_resolution          = 0.25
row_resolution             = 0.25

text_header = 'This is a sample text header.'

data = np.complex64(np.random.rand(100,100,4))

LSIWriter('test_lsi.lsi', pixels_per_column, pixels_per_row,channels,
          numeric_type_indicator, apodization_type, 
          remap_type, image_plane_indicator, rf_center_frequency, 
          rf_bandwidth, dwell_angle, cone_angle, graze_angle, twist_angle, 
          column_sample_spacing, row_sample_spacing, 
          column_oversampling_factor, row_oversampling_factor, 
          column_resolution, row_resolution,
          text_header, data)

data_read, meta_read = LSIReader('test_lsi.lsi')
