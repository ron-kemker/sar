# -*- coding: utf-8 -*-
'''
File: cv_example.py
Description: Example script for Civilian Vehicle Data Dome Dataset 
Author: Ronald Kemker

'''

import numpy as np
from fileIO.cvdata import CVData
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer
from signal_processing import spatial_variant_apodization as SVA
import matplotlib.pyplot as plt
from autofocus import phase_gradient_autofocus as PGA
from signal_processing import residual_video_phase_compensation as RVP_Comp

if __name__ == '__main__':
    data_path = '..\..\data\Civilian Vehicles\Domes\Camry\Camry_el30.0000.mat'
    target = data_path.split('\\')[-2]
    plt.close('all')
    
    timer = Timer('FileIO')
        
    with timer as _:
        sar_obj = CVData(data_path, target,  
                polarization='vv',
                center_frequency=9.6e9, 
                bandwidth=800e6,
                min_azimuth_angle = 40,
                max_azimuth_angle = 45,
                altitude = 15, # Meters
                )

    fig, ax = plt.subplots(1,3, figsize=[8,8])
    plt.tight_layout()
    
    sar_obj.cphd = RVP_Comp(sar_obj.cphd, sar_obj.freq)
    
    timer = Timer('Polar Format Algorithm')
    with timer as _:
        image = PFA(sar_obj, 
                single_precision=True,
                n_jobs=8,
                taylor_weighting=30,
                )
    
        imshow(image, ax=ax[0], dynamic_range=30)
        ax[0].axis('off')
        ax[0].title.set_text('Polar Format Algorithm')
        plt.tight_layout()
    
    timer = Timer('Spatially Variant Apodization')
    with timer as _:
        image_sva = SVA(image, N=1)
        imshow(image_sva, ax=ax[1], dynamic_range=30)
        ax[1].axis('off')
        ax[1].title.set_text('Spatially Variant Apodization')
        plt.tight_layout()
    
    timer = Timer('Phase Gradient Autofocus')
    with timer as _:
        image_af, ph_error = PGA(image_sva)
        imshow(image_af, ax=ax[2], dynamic_range=30)
        ax[2].axis('off')
        ax[2].title.set_text('Phase Gradient Autofocus')
        plt.tight_layout()

