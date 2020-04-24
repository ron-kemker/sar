# -*- coding: utf-8 -*-
'''
File: gotcha_example.py
Description: Example script for GOTCHA datasets. 
Author: Ronald Kemker

'''

from fileIO.gotcha import GOTCHA
from image_formation import polar_format_algorithm as PFA
from utils import imshow, Timer, polyphase_interp
from signal_processing import taylor_window
import numpy as np
import matplotlib.pyplot as plt
from autofocus import multi_aperture_map_drift_algorithm as MAM
from autofocus import phase_gradient_autofocus as PGA
from autofocus import spatial_variant_autofocus2 as SVA

data_path ='..\..\data\GOTCHA\DATA\pass1\VV'

timer = Timer('FileIO')
with timer as _:
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=650e6,
                taper_func=None,
                min_azimuth_angle=-50,
                max_azimuth_angle=-46,
                )

fig, ax = plt.subplots(2,2, figsize=[8,8])
txt = ['PFA',"PFA w/ Multi-Aperture Map-Drift",
        "PFA w/ Phase Gradient Autofocus"]
af = [None, MAM, PGA]

for i in range(len(txt)):

    timer = Timer(txt[i])
    with timer as _:
        image = PFA(sar_obj, 
                    upsample=False, 
                    interp_func = np.interp,
                    auto_focus = af[i],
                    single_precision=False,
                    )
    r = i // 2
    c = i % 2
    imshow(image.T, ax=ax[c,r])
    ax[c,r].axis('off')
    ax[c,r].title.set_text(txt[i])

    if i == 0:
        timer = Timer('PFA w/ Spatially Variant Autofocus')
        with timer as _:
            image_af = SVA(image)
        imshow(image_af.T, ax=ax[1,1])
        ax[1,1].axis('off')
        ax[1,1].title.set_text('PFA w/ Spatially Variant Autofocus')

