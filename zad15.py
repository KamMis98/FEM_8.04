# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:07:47 2021

@author: HOME
"""

import numpy as np
def lamda_15(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<0:
          y[i]=np.sin(x[i])
        else:
            y[i]=np.sqrt(x[i])
    return(y)