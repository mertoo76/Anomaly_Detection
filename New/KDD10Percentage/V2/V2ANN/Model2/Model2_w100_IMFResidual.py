# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:13:49 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder



#load x values
x_train=np.load("../../DataV2/Window100/xData.npy")
x_test=np.load("../../DataV2/testData/xDataTest.npy")



tmp=pd.read_csv('../../../Data/kddcup.data_10_percent_corrected',',')
y_train = tmp.iloc[:, 41].values

#labels
for i in range(y_train.shape[0]):
    if y_train[i] != 'normal.':
        #print(y_prep[i])
        y_train[i]='attack'
    
y_train=y_train[:x_train.shape[0]]

#1 normal, 0 attack
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

tmp=pd.read_csv('../../../Data/corrected',',')
y_test = tmp.iloc[:, 41].values


#labels Test

for i in range(y_test.shape[0]):
    if y_test[i] != 'normal.':
        #print(y_prep[i])
        y_test[i]='attack'
    
y_test=y_test[:x_test.shape[0]]
#1 normal, 0 attack
y_test = labelencoder_y.fit_transform(y_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 8))

#Adding a second hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#plot_model(classifier, show_shapes=True, to_file='../Model2_w100_IMFResidual.png')

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 20)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#the performance of the classification model
print("the Accuracy is: "+ str((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])))
recall = cm[1,1]/(cm[1,0]+cm[1,1]) #True positive/total actual positive
print("Recall is : "+ str(recall))
print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
precision = cm[1,1]/(cm[0,1]+cm[1,1])#True positive/predictive positive
print("Precision is: "+ str(precision))
print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))
from math import log
print("Entropy is: "+ str(-precision*log(precision)))
print("Confusion Matrix:\n",cm)