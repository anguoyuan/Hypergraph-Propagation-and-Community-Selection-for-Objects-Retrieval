# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:17:29 2021

@author: Guoyuan An
"""

import numpy as np

def connect_nodup(first_list, second_list):
    
    second_list=np.setdiff1d(second_list, first_list, assume_unique=True)
    
    return first_list+list(second_list)