# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:23:23 2018

@author: user
"""

# 4 -> src bytes , 5 -> dst bytes, 22 -> cpunt, 23-> srv_count, 31-> dst_host_count, 32-> dst_host_srv_count

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
x=np.zeros([2*x_prep.shape[1]*int(window-1),100])
for j in range(x_prep.shape[1]):
    i=0
    while i<window-1:
        #print('while---------',i,'\n')
        s=x_prep[i*100:100*(i+1),j]
        s=s.astype(int).reshape(s.size)
        t=np.arange(100*i*2,100*(i+1)*2,2)
        IMF = emd.emd(s,t,max_imf=1)
        x[j*int(window-1)+(i*2):(j*int(window-1))+((i+1)*2)]=IMF
        i=i+1


#np.save("xData", x)
#data=np.load("xData.npy")

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



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 100))

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

