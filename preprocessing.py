# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:05:25 2020
"""

#----------------Start--------------------------------

#preprocessing data
#scale data
from sklearn import preprocessing
import numpy as np

a = np.array([[10,2.7,3.6],
             [-100,5,-2],
             [120,20,40]], dtype=np.float64)

print(a)
print(preprocessing.scale(a))





#-------------------End------------------------------




