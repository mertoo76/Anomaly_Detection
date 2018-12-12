# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:41:09 2018

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

'''
#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
'''


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 4))

#Adding a second hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#plot_model(classifier, show_shapes=True, to_file='../model2.png')

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