# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:31:51 2018

@author: user
"""

from PyEMD import EMD
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

import pandas as pd


#Load dataset
kdd=pd.read_csv('./New/NSLKDD/Data/KDDTrain+.csv',',')

x=kdd.iloc[:,22:24]
x=np.append(x, kdd.iloc[:,31:33], axis=1)
y = kdd.iloc[:, 41].values

#labelencoder_y = LabelEncoder()
#yc = labelencoder_y.fit_transform(y)

feature_name=["count","srv_count","dst_host_count","dst_host_srv_count"]
#time
t=np.arange(0,x.shape[0]*2,2)
#Smurf(dos):startBorder=7750 - endBorder=8000 , startBorder=11375 - endBorder=11625 
#startBorder=50900 - endBorder=51000, startBorder=51950 - endBorder=52050
#ipsweep: startBorder=52130 - endBorder=52230, startBorder=52550 - endBorder=52650
startBorder=52130
endBorder=52230
for i in range(x.shape[1]):
    plt.plot(t[startBorder:endBorder],x[startBorder:endBorder,i],)
    j=0
    for data in y[startBorder:endBorder]:
        if data != y[0]:
            plt.scatter(t[startBorder+j], x[startBorder+j,i], c='r', label='data')
        j=j+1
    plt.xlabel("Time (s)")
    plt.ylabel(feature_name[i])
    plt.show()
    
#PCA
y_prep=kdd.target.copy()
normal=y_prep[0]
for i in range(len(y_prep)):
    if y_prep[i] != normal:
        #print(y_prep[i])
        y_prep[i]='attack'
    else:
        y_prep[i]='normal'
labelencoder_y = LabelEncoder()
yc = labelencoder_y.fit_transform(y_prep)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
projected = pca.fit_transform(x)
sc = plt.scatter(projected[:,0], projected[:,1], c = yc)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar(sc)


###
'''
#PCA
y_prep=kdd.target.copy()
normal=y_prep[0]
for i in range(len(y_prep)):
    if y_prep[i] != normal:
        #print(y_prep[i])
        y_prep[i]='attack'
    else:
        y_prep[i]='normal'
labelencoder_y = LabelEncoder()
yc = labelencoder_y.fit_transform(y_prep)

from sklearn.decomposition import PCA
types=set(kdd.target)
colour=np.arange(0,len(types),1)
yPCA=y.copy()
yPCA.map(types,colour)
pca = PCA(n_components=2)
projected = pca.fit_transform(x)
print(projected.shape)
plt.scatter(projected[:, 0], projected[:, 1],
            c=yc, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.legend(['Attack','Normal'])
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
'''
