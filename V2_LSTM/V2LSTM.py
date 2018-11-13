# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:37:45 2018

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
x_prep=kdd.data[:,4:6]
x_prep=np.append(x_prep, kdd.data[:,22:24], axis=1)
x_prep=np.append(x_prep, kdd.data[:,31:33], axis=1)

#IMFs already extracted, you can skip this part and start with load x data
#EMD -> extract IMF
emd=EMD()

window=x_prep.shape[0]/100
window=window-1
x=np.zeros([6*2,x_prep.shape[0]]) #feature*IMF count
for j in range(x_prep.shape[1]):
    i=0
    print(j," Feature\n")
    while i<window:
        #print('while---------',i,'\n')
        s=x_prep[i*100:100*(i+1),j]
        s=s.astype(int).reshape(s.size)
        t=np.arange(100*i*2,100*(i+1)*2,2)
        IMF = emd.emd(s,t,max_imf=1)
        x[j*2:(j*2)+2,i*100:(i*100)+100]=IMF
        i=i+1
x=x.transpose()
newLen=i*100
x=x[:newLen]

#np.save("xData", x)
#data=np.load("xData.npy")

#x shape[0]= total window*IMF*feature= 4939*2*6
x=np.load("xData.npy")

#labels
y=kdd.target.copy()
normal=y[0]
for i in range(x.shape[0]):
    if y[i] != normal:
        #print(y_prep[i])
        y[i]='attack'
    else:
        y[i]='normal'
y=y[:x.shape[0]]
#1 normal, 0 attack
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding, LSTM, Dropout

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

model = Sequential()
model.add(LSTM(64,batch_input_shape=(None,12,1),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid',init = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size = 12, nb_epoch = 20)


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