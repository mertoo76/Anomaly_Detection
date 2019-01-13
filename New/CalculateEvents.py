# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:32:38 2018

@author: user
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Train Part#############
#importing the dataset
dataset = pd.read_csv('./KDD10Percentage/Data/kddcup.data_10_percent_corrected',',')
#dataset = pd.read_csv('./KDD10Percentage/Data/corrected',',')

y_train = dataset.iloc[:, 41].values


'''
#11 = normal
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
'''

i=0
eventsCount=0
while i<y_train.shape[0]-1:
    if (y_train[i] != y_train[i+1]) and (y_train[i+1]!='normal.'):
        eventsCount = eventsCount + 1
    i= i + 1


#status Change
i=0
eventsCount=0
while i<y_train.shape[0]-1:
    if (y_train[i] != y_train[i+1]):
        eventsCount = eventsCount + 1
    i= i + 1
