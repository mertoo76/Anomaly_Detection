# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:04:01 2019

@author: user
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Train Part#############
#importing the dataset
dataset = pd.read_csv('../../Data/kddcup.data_10_percent_corrected',',')
#change Multi-class to binary-class
#dataset['normal'] = dataset['normal'].replace(['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'], 'attack')

x_train=dataset.iloc[:,22:24]
x_train=np.append(x_train, dataset.iloc[:,31:33], axis=1)
y_train = dataset.iloc[:, 41].values

for i in range(len(y_train)):
    if y_train[i] != 'normal.':
        #print(y_prep[i])
        y_train[i]='attack'
    

#1 normal, 0 attack
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

#Test Part ################################
#importing the dataset
datasetTest = pd.read_csv('../../Data/corrected',',')
#change Multi-class to binary-class
#datasetTest['neptune'] = datasetTest['neptune'].replace(['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'], 'attack')

x_test=datasetTest.iloc[:,22:24]
x_test=np.append(x_test, datasetTest.iloc[:,31:33], axis=1)
y_test = datasetTest.iloc[:, 41].values


for i in range(len(y_test)):
    if y_test[i] != 'normal.':
        #print(y_prep[i])
        y_test[i]='attack'
    

y_test = labelencoder_y.fit_transform(y_test)


#prepare input for LSTM
batch_size=10
y_train =y_train[:int(y_train.shape[0]/batch_size)*batch_size]
y_train = y_train.reshape((int(y_train.shape[0]/batch_size),batch_size))

x_train =x_train[:int(x_train.shape[0]/batch_size)*batch_size]
x_train = x_train.reshape((int(x_train.shape[0]/batch_size),batch_size,x_train.shape[1]))

y_test =y_test[:int(y_test.shape[0]/batch_size)*batch_size]
y_test = y_test.reshape((int(y_test.shape[0]/batch_size),batch_size))

x_test =x_test[:int(x_test.shape[0]/batch_size)*batch_size]
x_test = x_test.reshape((int(x_test.shape[0]/batch_size),batch_size,x_test.shape[1]))

#convert label shape one label per sequence
tmp=y_train.copy()
y_train=np.ones((y_train.shape[0],1))

for i in range(tmp.shape[0]):
    if sum(tmp[i]) < 9:
        y_train[i][0]=0


tmp=y_test.copy()
y_test=np.ones((y_test.shape[0],1))

for i in range(tmp.shape[0]):
    if sum(tmp[i]) < 9:
        y_test[i][0]=0


# Importing the Keras libraries and packages
#####################################################################################################
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils import plot_model
model = Sequential()
model.add(LSTM(32,batch_input_shape=(None,10,4),return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid',init = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
#plot_model(model, show_shapes=True, to_file='../model1.png')

model.fit(x_train, y_train, nb_epoch = 20)


# Predicting the Test set results
y_pred = model.predict(x_test)
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