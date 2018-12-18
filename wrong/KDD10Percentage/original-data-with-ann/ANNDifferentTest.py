# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:55:28 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


#Load dataset
kdd=datasets.fetch_kddcup99(percent10=True)

#Prepare input signal for EMD
x_train=kdd.data[:,4:6]
x_train=np.append(x_train, kdd.data[:,22:24], axis=1)
x_train=np.append(x_train, kdd.data[:,31:33], axis=1)

#labels
y_train=kdd.target.copy()
normal=y_train[0]
for i in range(len(y_train)):
    if y_train[i] != normal:
        #print(y_prep[i])
        y_train[i]='attack'
    else:
        y_train[i]='normal'
#1 normal, 0 attack
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

testData= pd.read_csv('../testData',',')
y_test=testData.iloc[:, 41].values
x_test = testData.iloc[:, 4:6].values
x_test=np.append(x_test, testData.iloc[:,22:24], axis=1)
x_test=np.append(x_test, testData.iloc[:,31:33], axis=1)

#labels Test

for i in range(y_test.shape[0]):
    if y_test[i] != 'normal':
        #print(y_prep[i])
        y_test[i]='attack'
    
y_test=y_test[:x_test.shape[0]]
#1 normal, 0 attack
y_test = labelencoder_y.fit_transform(y_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 6))

#Adding a second hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

#Adding a third hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

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
recall = cm[1,1]/(cm[0,1]+cm[1,1])
print("Recall is : "+ str(recall))
print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
precision = cm[1,1]/(cm[1,0]+cm[1,1])
print("Precision is: "+ str(precision))
print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))
from math import log
print("Entropy is: "+ str(-precision*log(precision)))
