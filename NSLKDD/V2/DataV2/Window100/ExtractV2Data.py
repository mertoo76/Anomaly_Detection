# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 18:50:09 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt
import pandas as pd



#Load dataset
kdd= pd.read_csv('../../../Data/OriginalData/KDDTrain+.csv',',')

#Prepare input signal for EMD
x_prep=kdd.iloc[:,4:6]
x_prep=np.append(x_prep, kdd.iloc[:,22:24], axis=1)
x_prep=np.append(x_prep, kdd.iloc[:,31:33], axis=1)

#IMFs already extracted, you can skip this part and start with load x data
#EMD -> extract IMF
emd=EMD()

window=x_prep.shape[0]/100
window=window-1
x=np.zeros([6*2,x_prep.shape[0]]) #feature*IMF count
for j in range(x_prep.shape[1]):
    i=0
    print(j," Feature\n")
    while i<window:
        #print('while---------',i,'\n')
        s=x_prep[i*100:100*(i+1),j]
        s=s.astype(int).reshape(s.size)
        t=np.arange(100*i*2,100*(i+1)*2,2)
        IMF = emd.emd(s,t,max_imf=1)
        x[j*2:(j*2)+2,i*100:(i*100)+100]=IMF
        i=i+1
x=x.transpose()
newLen=i*100
x=x[:newLen]

np.save("xData", x)