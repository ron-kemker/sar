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

data_path ='..\..\data\GOTCHA\DATA\pass8\HH'

timer = Timer('FileIO')
with timer as _:
    sar_obj = GOTCHA(data_path,
                center_frequency=9.6e9, 
                bandwidth=300e6,
                taper_func=taylor_window,
                min_azimuth_angle=40,
                max_azimuth_angle=42
                )

fig, ax = plt.subplots(1, 3, figsize=[12,10])
txt = ['PFA',"PFA w/ Multi-Aperture Map-Drift",
       "PFA w/ Phase Gradient Autofocus"]
af = [None, MAM, PGA]

for i in range(len(txt)):

    timer = Timer(txt[i])
    with timer as _:
        image = PFA(sar_obj, 
                    upsample=False, 
                    interp_func = np.interp,
                    auto_focus = af[i]
                    )

    imshow(image, ax=ax[i])
    ax[i].axis('off')
    ax[i].title.set_text(txt[i])

plt.tight_layout()