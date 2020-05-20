# -*- coding: utf-8 -*-
'''

File: sandia.py
Description: Contains software to process Snadia SAR data
Author: Ronald Kemker

'''
from scipy.io import loadmat
import numpy as np
import warnings
from glob import glob
import os
from fnmatch import fnmatch
from numpy.linalg import norm

class SANDIA(object):
    """Processes the Sandia Data
    
    @author: Ronald Kemker

    This code snippet will load and display the data dome data:
    data_path = 'data\Civilian Vehicles\Domes\Camry\Camry_el40.0000.mat'
    cvdata = CVData(data_path, 'Camry')
    cvdata.imshow()

    # Arguments
        data_path: String. File path to desired data file (.mat)
        single_precision: Boolean.  If false, it will be double precision.
    
    # References
        -TBD
    """
    def __init__(self, data_path, single_precision=True):
                
        if single_precision:
            fdtype = np.float32
            cdtype = np.complex64
        else:
            fdtype = np.float64
            cdtype = np.complex128           
        
        c = fdtype(299792458) # Speed of Light
        pi = np.pi

        #get filename containing auxilliary data
        for file in os.listdir(data_path):
                if fnmatch(file, '*.au2'):
                    aux_fname = data_path+file    
        
        #import auxillary data
        f=open(aux_fname,'rb')
        
        #initialize tuple
        record=['blank'] #first record blank to ensure
                         #indices match record numbers
        
        #record 1
        data = np.fromfile(f, dtype = np.dtype([
            ('version','S6'),
            ('phtype','S6'),
            ('phmode','S6'),
            ('phgrid','S6'),
            ('phscal','S6'),
            ('cbps','S6')
            ]),count=1)
        record.append(data[0])
        
        #record 2
        f.seek(44)
        data = np.fromfile(f, dtype = np.dtype([
            ('npulses','i4'),
            ('nsamples','i4'),
            ('ipp_start','i4'),
            ('ddas','f4',(5,)),
            ('kamb','i4')
            ]),count=1)
        record.append(data[0])
        
        #record 3    
        f.seek(44*2)
        data = np.fromfile(f, dtype = np.dtype([
            ('fpn','f4',(3,)),
            ('grp','f4',(3,)),
            ('cdpstr','f4'),
            ('cdpstp','f4')
            ]),count=1)
        record.append(data[0])
        
        #record 4
        f.seek(44*3)
        data = np.fromfile(f, dtype = np.dtype([
            ('f0','f4'),
            ('fs','f4'),
            ('fdot','f4'),
            ('r0','f4')
            ]),count=1)
        record.append(data[0])
        
        #record 5 (blank)rvr_au_read.py
        f.seek(44*4)
        data = []
        record.append(data)
        
        #record 6
        npulses = record[2]['npulses']
        rpoint = np.zeros([npulses,3])
        deltar = np.zeros([npulses,])
        fscale = np.zeros([npulses,])
        c_stab = np.zeros([npulses,3])
        #build up arrays for record(npulses+6)
        for n in range(npulses):
            f.seek((n+5)*44)
            data = np.fromfile(f, dtype = np.dtype([
                ('rpoint','f4',(3,)),
                ('deltar','f4'),
                ('fscale','f4'),
                ('c_stab','f8',(3,))
                ]),count=1)
            rpoint[n,:] = data[0]['rpoint']
            deltar[n] = data[0]['deltar']
            fscale[n] = data[0]['fscale']
            c_stab[n,:] = data[0]['c_stab']
        #consolidate arrays into a 'data' dataype
        dt = np.dtype([
                ('rpoint','f4',(npulses,3)),
                ('deltar','f4',(npulses,)),
                ('fscale','f4',(npulses,)),
                ('c_stab','f8',(npulses,3))
                ])        
        data = np.array((rpoint,deltar,fscale,c_stab)
                ,dtype=dt)
        #write to record file
        record.append(data)
        
        #import phase history
        for file in os.listdir(data_path):
            if fnmatch(file, '*.phs'):
                phs_fname = data_path+file
                
        K = record[2][1]
        Np = record[2][0]
        
        f=open(phs_fname,'rb')    
        dt = np.dtype('i2')
            
        phs  = np.fromfile(f, dtype=dt, count=-1)
        real = phs[0::2].reshape([Np,K])  
        imag = phs[1::2].reshape([Np,K])
        phs  = real+1j*imag
        
        delta_t   = record[4]['fs']
        t         = np.linspace(-K/2, K/2, K)*1.0/delta_t
        chirprate = record[4]['fdot']*1.0/(2*pi)
        f_0       = record[4]['f0']*1.0/(2*pi) + chirprate*K/(2*delta_t)
        B_IF      = (t.max()-t.min())*chirprate
        delta_r   = c/(2*B_IF)
        freq      = f_0+chirprate*t
        
        self.n_hat = record[3]['fpn']
        self.cphd = cdtype(phs)
        self.num_pulses = Np
        self.num_samples = K
        self.freq = freq
        self.antenna_location = record[6]['rpoint'].T
        self.k_r = 4*pi*freq/c
        self.delta_r = delta_r
        self.chirprate = chirprate
        self.bandwidth = B_IF
        self.f_0 = f_0
        