# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:01:30 2018

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
x=kdd.data[:,22:24]
x=np.append(x, kdd.data[:,31:33], axis=1)

#labels
y_prep=kdd.target.copy()
normal=y_prep[0]
for i in range(len(y_prep)):
    if y_prep[i] != normal:
        #print(y_prep[i])
        y_prep[i]='attack'
    else:
        y_prep[i]='normal'
#1 normal, 0 attack
labelencoder_y = LabelEncoder()
y_prep = labelencoder_y.fit_transform(y_prep)

#prepare input for LSTM
batch_size=100
y =y_prep[:int(y_prep.shape[0]/batch_size)*batch_size]
y = y.reshape((int(y.shape[0]/batch_size),batch_size))

x =x[:int(x.shape[0]/batch_size)*batch_size]
x = x.reshape((int(x.shape[0]/batch_size),batch_size,x.shape[1]))


#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


#####################################################################################################
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils import plot_model


model = Sequential()
model.add(LSTM(64,batch_input_shape=(None,100,4),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(100,activation='sigmoid',init = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

plot_model(model, show_shapes=True, to_file='model.png')


model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)


# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

y_pred=y_pred.reshape((y_pred.shape[0]*y_pred.shape[1]))
y_test=y_test.reshape((y_test.shape[0]*y_test.shape[1]))


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