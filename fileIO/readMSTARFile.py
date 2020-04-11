# -*- coding: utf-8 -*-
'''

File: readMSTARFile.py
Description: This parses the MSTAR file and passes back the complex image, 
             truth label, serial number, and depression angle.

Author: Ronald Kemker

'''

import numpy as np

def readMSTARFile(filename):
    """Processes MSTAR data files.
    
    @author: Ronald Kemker

    # Arguments
             filename: String. Path to the datafile to be processed.
        
    # Outputs
                  img: Float32.  NxMx2 image containing the amplitude and r
                       phase data fo the associated MSTAR file.  NxM is the
                       number of image pixels.
                label: String. This is a truth label of the target present in  
                       the image. This can be used for machine learning.
         targetSerNum: String. Serial number can be used to learn more info
                       about the target in the image.
     depression_angle: float. This is the depression angle of the collected
                                 image.
    
    # References
        - [MSTAR Overview](
          https://www.sdms.afrl.af.mil/index.php?collection=mstar)
    """    
    f = open(filename, 'rb')
    
    line = ''
    
    while 'EndofPhoenixHeader' not in line:
        line = str(f.readline())
        if 'TargetType' in line:
            label = line.split('=')[1].strip().split('\\n')[0]
        elif 'TargetSerNum' in line:
            targetSerNum = line.split('=')[1].strip().split('\\n')[0]
        elif 'NumberOfColumns' in line:
            cols = int(line.split('=')[1].strip().split('\\n')[0])
        elif 'NumberOfRows' in line:
            rows = int(line.split('=')[1].strip().split('\\n')[0])
        elif 'DesiredDepression' in line:
            depression_angle = int(line.split('=')[1].strip().split('\\n')[0])   
    
    data = np.fromfile(f, dtype='>f4')
    pix = rows*cols
    img = np.zeros((rows, cols , 2) , np.float32)
    img[:,:,0] = data[:pix].reshape(rows,cols) # Magnitude Data
    img[:,:,1] = data[pix:].reshape(rows,cols) # Phase Data
               
    f.close()
    return img, label, targetSerNum, depression_angle