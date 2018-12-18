# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:29:33 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder



#load x values
x_train=np.load("../DataV2/Window100/xData.npy")
x_test=np.load("../DataV2/Window100/xDataTest.npy")

x_train=x_train[:,4:]
x_test=x_test[:,4:]


tmp=pd.read_csv('../../Data/OriginalData/KDDTrain+.csv',',')
y_train = tmp.iloc[:, 41].values

#labels
for i in range(y_train.shape[0]):
    if y_train[i] != 'normal':
        #print(y_prep[i])
        y_train[i]='attack'
    
y_train=y_train[:x_train.shape[0]]

#1 normal, 0 attack
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

tmp=pd.read_csv('../../Data/OriginalData/KDDTest+.csv',',')
y_test = tmp.iloc[:, 41].values


#labels Test

for i in range(y_test.shape[0]):
    if y_test[i] != 'normal':
        #print(y_prep[i])
        y_test[i]='attack'
    
y_test=y_test[:x_test.shape[0]]
#1 normal, 0 attack
y_test = labelencoder_y.fit_transform(y_test)


#prepare input for LSTM
batch_size=100
y_train =y_train[:int(y_train.shape[0]/batch_size)*batch_size]
y_train = y_train.reshape((int(y_train.shape[0]/batch_size),batch_size))

x_train =x_train[:int(x_train.shape[0]/batch_size)*batch_size]
x_train = x_train.reshape((int(x_train.shape[0]/batch_size),batch_size,x_train.shape[1]))

y_test =y_test[:int(y_test.shape[0]/batch_size)*batch_size]
y_test = y_test.reshape((int(y_test.shape[0]/batch_size),batch_size))

x_test =x_test[:int(x_test.shape[0]/batch_size)*batch_size]
x_test = x_test.reshape((int(x_test.shape[0]/batch_size),batch_size,x_test.shape[1]))

def load_Model():
    from keras.models import model_from_yaml
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model
    
def save_Model(model):
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding, LSTM, Dropout
from keras.utils import plot_model

model = Sequential()
model.add(LSTM(64,batch_input_shape=(None,100,8),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(100,activation='sigmoid',init = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()



model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)

#Save model
save_Model(model)

#Load Model
model=load_Model()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

y_pred=y_pred.reshape((y_pred.shape[0]*y_pred.shape[1]))
y_test=y_test.reshape((y_test.shape[0]*y_test.shape[1]))


#Event Prediction
total_event_change=0
for i in range(len(y_test)-1):
    if y_test[i]!=y_test[i+1]:
        total_event_change= total_event_change + 1

def checkChange(correct_event_change,i,y_test,error):
    j=i-error
    end=i+error
    if j<0:
        j=0
    if end > len(y_test):
        end=len(y_test)
    while (j<end) & (j!=len(y_test)):
        if y_test[j] != y_test[j+1]:
            correct_event_change = correct_event_change + 1
            return correct_event_change
        j=j+1
    return correct_event_change

correct_event_change=0
for i in range(len(y_test)-1):
    if y_pred[i] != y_pred[i+1]:
        correct_event_change=checkChange(correct_event_change,i,y_test,2)

accuracy=correct_event_change/total_event_change
print("Event Detection Accuracy=",accuracy)

