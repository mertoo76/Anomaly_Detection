# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 18:50:09 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


#Load dataset
kdd=datasets.fetch_kddcup99(percent10=True)

#Prepare input signal for EMD
x_prep=kdd.data[:,4:6]
x_prep=np.append(x_prep, kdd.data[:,22:24], axis=1)
x_prep=np.append(x_prep, kdd.data[:,31:33], axis=1)

#IMFs already extracted, you can skip this part and start with load x data
#EMD -> extract IMF
emd=EMD()

window=x_prep.shape[0]/1000
window=window-1
x=np.zeros([6*2,x_prep.shape[0]]) #feature*IMF count
for j in range(x_prep.shape[1]):
    i=0
    print(j," Feature\n")
    while i<window:
        #print('while---------',i,'\n')
        s=x_prep[i*1000:1000*(i+1),j]
        s=s.astype(int).reshape(s.size)
        t=np.arange(1000*i*2,1000*(i+1)*2,2)
        IMF = emd.emd(s,t,max_imf=1)
        x[j*2:(j*2)+2,i*1000:(i*1000)+1000]=IMF
        i=i+1
x=x.transpose()
newLen=i*1000
x=x[:newLen]

np.save("xData1000", x)