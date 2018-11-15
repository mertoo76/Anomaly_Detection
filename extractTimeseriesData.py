# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:21:58 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


#Load dataset
kdd=datasets.fetch_kddcup99(percent10=True)


x_prep = t=np.arange(0,kdd.data.shape[0]*2,2)
x_prep=x_prep[:,np.newaxis]
x_prep=np.append(x_prep, kdd.data[:,4:6], axis=1)
x_prep=np.append(x_prep, kdd.data[:,22:24], axis=1)
x_prep=np.append(x_prep, kdd.data[:,31:33], axis=1)

np.save("timeSeries", x_prep)

