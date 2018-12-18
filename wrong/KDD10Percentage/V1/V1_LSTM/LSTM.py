# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:40:07 2018

@author: user
"""

from PyEMD import EMD
import numpy as np
import pylab as plt

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


#Load dataset
kdd=datasets.fetch_kddcup99(percent10=True)

#Prepare input signal for EMD
x_prep=kdd.data[:,4:6]
x_prep=np.append(x_prep, kdd.data[:,22:24], axis=1)
x_prep=np.append(x_prep, kdd.data[:,31:33], axis=1)


#x shape[0]= total window*IMF*feature= 4939*2*6
x=np.load("xData.npy")
window=x_prep.shape[0]/100
#get target data
i=0
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


y=np.ones([2*x_prep.shape[1]*int(window-1),1],dtype=int)
i=0
while i<window-1:
    att=1
    for k in range(i*100,(i+1)*100):
        #whic means if attack
        if y_prep[k] == 0:
            att=0
    for j in range(x_prep.shape[1]):
        y[j*int(window-1)+(i*2):(j*int(window-1))+((i+1)*2)]=att           
    i=i+1
                
                
        
#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
   
#####################################################################################################
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

model = Sequential()
model.add(LSTM(64,batch_input_shape=(None,100,1),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid',init = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size = 100, nb_epoch = 20)


# Predicting the Test set results
y_pred = model.predict(x_test)
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

'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
'''
'''
# load YAML and create model
from keras.models import model_from_yaml
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Predicting the Test set results
y_pred = loaded_model.predict(x_test)
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
'''