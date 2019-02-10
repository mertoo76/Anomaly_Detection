# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:27:44 2019

@author: user
"""

import numpy as np
import pylab as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder





#Load V2 data with window size 100
#x shape[0]= total window*IMF*feature= 4939*2*6
#load x values
x_train=np.load("../Data/xData.npy")
x_test=np.load("../Data/xDataTest.npy")

#labels

tmp=pd.read_csv('../../New/KDD10Percentage/Data/kddcup.data_10_percent_corrected',',')
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

#Test 
tmp=pd.read_csv('../../New/KDD10Percentage/Data/corrected',',')
y_test = tmp.iloc[:, 41].values


#labels Test

for i in range(y_test.shape[0]):
    if y_test[i] != 'normal.':
        #print(y_prep[i])
        y_test[i]='attack'
    
y_test=y_test[:x_test.shape[0]]
#1 normal, 0 attack
y_test = labelencoder_y.fit_transform(y_test)



#prepare input for LSTM
batch_size=10
#y =y_prep[:int(y_prep.shape[0]/batch_size)*batch_size]
y_train = y_train.reshape((int(y_train.shape[0]/batch_size),batch_size))
y_test = y_test.reshape((int(y_test.shape[0]/batch_size),batch_size))

x_train = x_train.reshape((int(x_train.shape[0]/batch_size),batch_size,x_train.shape[1]))
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
model.add(LSTM(32,batch_input_shape=(None,10,8),return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(10,activation='sigmoid',init = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

#plot_model(model, show_shapes=True, to_file='../Model3_w100_IMFResidual.png')

model.fit(x_train, y_train, nb_epoch = 20)


#Save model
#save_Model(model)

#Load Model
model=load_Model()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

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
recall = cm[1,1]/(cm[1,0]+cm[1,1]) #True positive/total actual positive
print("Recall is : "+ str(recall))
print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
precision = cm[1,1]/(cm[0,1]+cm[1,1])#True positive/predictive positive
print("Precision is: "+ str(precision))
print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))
from math import log
print("Entropy is: "+ str(-precision*log(precision)))
print("Confusion Matrix:\n",cm)