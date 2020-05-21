# -*- coding: utf-8 -*-
'''
File: polarization_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow
import matplotlib.pyplot as plt
from signal_processing import residual_video_phase_compensation as RVP_Comp
from signal_processing import spatial_variant_apodization as SVA
from autofocus import phase_gradient_autofocus as PGA
from utils import pauli_decomposition as PD

if __name__ == '__main__':

    data_path ='..\..\data\GOTCHA\DATA\pass1\\'
    dynamic_range = 30
    plt.close('all')
        
    fig, ax = plt.subplots(2,2 , figsize=[8,8])
    polarization = ['HH','VV','HV']
    image_arr = []
    for i, pol in enumerate(polarization):
        
        sar_obj = GOTCHA(data_path+pol,
                    center_frequency=9.6e9, 
                    bandwidth=650e6,
                    min_azimuth_angle = 178,
                    max_azimuth_angle = 182,
                    verbose=False,
                    )      
        
        sar_obj.cphd = RVP_Comp(sar_obj.cphd, sar_obj.freq)
        
        image = PFA(sar_obj, 
                single_precision=True,
                n_jobs=8,
                taylor_weighting = 30,
                )
        
        # image = SVA(image, N=1)
        # image, _ = PGA(image)
        
        imshow(image, dynamic_range, ax=ax[int(i//2)][int(i%2)])
        ax[int(i//2)][int(i%2)].axis('off')
        ax[int(i//2)][int(i%2)].title.set_text(pol)
        image_arr += [image]

    pauli_rgb = PD(image_arr[0],image_arr[1],image_arr[2])
    
    ax[1][1].imshow(pauli_rgb)    
    ax[1][1].axis('off')
    ax[1][1].title.set_text('False Color')