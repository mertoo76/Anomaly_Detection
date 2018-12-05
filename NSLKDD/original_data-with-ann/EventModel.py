# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:27:03 2018

@author: user
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Train Part#############
#importing the dataset
dataset = pd.read_csv('../Data/OriginalData/KDDTrain+.csv',',')
#change Multi-class to binary-class
#dataset['normal'] = dataset['normal'].replace(['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'], 'attack')

x_train=dataset.iloc[:,22:24]
x_train=np.append(x_train, dataset.iloc[:,31:33], axis=1)
y_train = dataset.iloc[:, 41].values

for i in range(len(y_train)):
    if y_train[i] != 'normal':
        #print(y_prep[i])
        y_train[i]='attack'
    

#1 normal, 0 attack
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

#Test Part ################################
#importing the dataset
datasetTest = pd.read_csv('../Data/OriginalData/KDDTest+.csv',',')
#change Multi-class to binary-class
#datasetTest['neptune'] = datasetTest['neptune'].replace(['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'], 'attack')

x_test=datasetTest.iloc[:,22:24]
x_test=np.append(x_test, datasetTest.iloc[:,31:33], axis=1)
y_test = datasetTest.iloc[:, 41].values


for i in range(len(y_test)):
    if y_test[i] != 'normal':
        #print(y_prep[i])
        y_test[i]='attack'
    

y_test = labelencoder_y.fit_transform(y_test)

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
from keras.layers import Dense
from keras.utils import plot_model

#Model already Saved You can skip this part.
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 4))

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

#Save model
save_Model(classifier)

#Load Model
classifier=load_Model()
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

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