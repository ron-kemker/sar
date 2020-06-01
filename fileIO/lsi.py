# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:52:19 2020

@author: Master
"""

import numpy as np
import struct
    
def LSIWriter(filename, pixels_per_column, pixels_per_row, channels, 
             numeric_type_indicator, apodization_type, 
             remap_type, image_plane_indicator, rf_center_frequency, 
             rf_bandwidth, dwell_angle, cone_angle, graze_angle, twist_angle, 
             column_sample_spacing, row_sample_spacing, 
             column_oversampling_factor, row_oversampling_factor, 
             column_resolution, row_resolution,
             text_header, data):
    """This code will write Laboratory SAR Image (.LSI) files 
    
    @author: Ronald Kemker

    This code will write Laboratory SAR Image (.LSI) files 
    (used for storing synthetic SAR data for ATR applications).

    # Arguments
        filename                   : string
                                     The file path to store the .LSI file
        pixels_per_column          : int32
                                     Number of x-axis pixels
        pixels_per_row             : int32
                                     Number of y-axis pixels
        channels                   : int32
                                     The number of image channels
        numeric_type_indicator     : int32 
                                     data type, 1=float32 or 2=complex64
        apodization_type           : int32 
                                     (0=None, other codes TBD)
        remap_type                 : int32 
                                     Pixel-wise remapping (0=None, 
                                     other codes TBD)
        image_plane_indicator      : int32 
                                     (1=slant, 2=equalized slant, 3=ground)
        rf_center_frequency        : float32
                                     The center frequency (Hz)
        rf_bandwidth               : float32
                                     The instantaneous bandwidth (Hz)
        dwell_angle                : float32
                                     The dwell angle is...
        cone_angle                 : float32
                                     Doppler-cone angle. This is the azimuth
                                     extent of the data. Also computed as the
                                     angle between the 1) Aperture Reference 
                                     Point - velocity (VARP) at time_center-
                                     of-aperture (COA) and 2) unit-vector 
                                     uLOS_coa.    
        graze_angle                : float32
                                     The grazing angle is the angle between 
                                     the earth tangent plane (ETP) x-axis and
                                     the slant plane x-axis.
        twist_angle                : float32
                                     The twist/tile angle is the angle between
                                     the earth tangent plane (ETP) y-axis and 
                                     the slant plane y-axis.
        column_sample_spacing      : float32
                                     Column-direction sample spacing (meters)
        row_sample_spacing         : float32
                                     Row-direction sample spacing (meters)
        column_oversampling_factor : float32
                                     Column-Direction Oversampling Factor
        row_oversampling_factor    : float32
                                     Row-Direction Oversampling Factor
        column_resolution          : float32
                                     Column-Direction resolution (meters)
        row_resolution             : float32
                                     Row-Direction resolution (meters)
        text_header                : string 
                                     Space-delimited contiguous character
                                     string, up to 200 characters.
        data                       : Numeric (float32 or complex64)
                                     The data to be saved
    
    # References
        Aerospace Coroporation LSI Description and Format
    """    
    
    file = open(filename, 'wb')
    
    # Write Int32 Header Values
    file.write(np.int32(pixels_per_column))
    file.write(np.int32(pixels_per_row))
    file.write(np.int32(channels))
    file.write(np.int32(numeric_type_indicator))
    file.write(np.int32(apodization_type))
    file.write(np.int32(remap_type))
    file.write(np.int32(image_plane_indicator))    
    
    # Write Float32 Header Values
    file.write(np.float32(rf_center_frequency))
    file.write(np.float32(rf_bandwidth))
    file.write(np.float32(dwell_angle))
    file.write(np.float32(cone_angle))
    file.write(np.float32(graze_angle))
    file.write(np.float32(twist_angle))
    file.write(np.float32(column_sample_spacing)) 
    file.write(np.float32(row_sample_spacing))
    file.write(np.float32(column_oversampling_factor))
    file.write(np.float32(row_oversampling_factor))
    file.write(np.float32(column_resolution))
    file.write(np.float32(row_resolution))
    
    file.write(bytes('\0' * (200-file.tell()),'utf-8'))

    # Exactly 200 characters 
    file.write(bytes(text_header[:200].ljust(200), 'utf-8'))
    
    if numeric_type_indicator == 1:
        file.write(np.float32(data))
    elif numeric_type_indicator == 2:
        file.write(np.complex64(data))
    else:
        err = 'Invalid "numeric_type_indicator".  Valid range is 1 or 2'
        ValueError(err)
        
    file.close()
    
def LSIReader(filename):
    """This code will read Laboratory SAR Image (.LSI) files 
    
    @author: Ronald Kemker

    This code will read Laboratory SAR Image (.LSI) files 
    (used for storing synthetic SAR data for ATR applications).

    # Arguments
        filename : string
                   The path of the .LSI file to read
    # Outputs
        data     : Numpy array 
                   Containing SAR data (float32 or complex64)
        metadata : dictionary
                   Contains all of the relevant metadata listed in LSIWriter
    # References
        Aerospace Coroporation LSI Description and Format
    """   

    metadata = {}
    
    dict_args_int32 = ['pixels_per_column','pixels_per_row','channels',
    'numeric_type_indicator','apodization_type','remap_type',
    'image_plane_indicator']
    
    dict_args_float32 = ['rf_center_frequency','rf_bandwidth','dwell_angle',
    'cone_angle','graze_angle','twist_angle','column_sample_spacing',
    'row_sample_spacing','column_oversampling_factor',
    'row_oversampling_factor','column_resolution','row_resolution']

    file = open(filename, "rb")
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0, 0)
    num = file.read(200)
    text = file.read(200)
    data = file.read(file_size - file.tell())
    file.close()
    
    for i, arg in enumerate(dict_args_int32):
        metadata[arg] = np.int32(struct.unpack('<i', num[4*i:4*i+4]))

    N = len(dict_args_int32) * 4
    for i, arg in enumerate(dict_args_float32):
        metadata[arg] = np.float32(struct.unpack('<f', num[N+4*i:4*i+4+N]))
        
    metadata['text_header'] = str(text, 'utf-8')
    
    
    if metadata['numeric_type_indicator'][0] == 1:
        data = np.frombuffer(data, np.float32)
    elif metadata['numeric_type_indicator'][0] == 2:
        data = np.frombuffer(data, np.complex64)
    else:
        err = 'Invalid "numeric_type_indicator".  Valid range is 1 or 2'
        ValueError(err)  
    
    data = data.reshape(metadata['pixels_per_row'][0], 
                        metadata['pixels_per_column'][0],
                        metadata['channels'][0])
        
    return data, metadata