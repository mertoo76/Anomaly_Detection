# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:09:40 2018

@author: user
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
import collections
import munkres

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

#plot_model(model, show_shapes=True, to_file='../model1.png')

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


def evaluation_window_adp(fact, detection, window=0, return_match=False):

    """ a variation of evaluation_window() which is adapted to sparse cost matrix generated from fact and detection.



    If fact or detection contain many elements, say more than one hundred. It will take a significant amount of time,

    even with hungarian algo, to compute the min cost maximum matching.

    In our specific case, since the cost matrix is very specific, and can only have values at limited places.

    It is thus possible to cut the initial cost matrix into several non-connecting ones. For example:

    cost_matrix = [[62, 0,  0,  0, 0,  0, 0],

                   [11, 11, 82, 0, 0,  0, 0],

                   [0,  0, 81, 12, 0,  0, 0],

                   [0,  0,  0,  0, 12, 0, 0],

                   [0,  0,  0,  0, 0,  0, 0],

                   [0,  0,  0,  0, 0,  0, 12],

                   [0,  0,  0,  0, 0,  0, 12]]

    The given cost matrix is composed of three separate parts:

    cost_matrix[0:4][0:5], cost_matrix[3:4][4:5] and cost_matrix[5:end][6:end].

    Calculating the matching separately for the two sub-matrices will be faster.



    Args:

        fact (list of int): the index or timestamp of facts/events to be detected

        detection (list of int): index or timestamp of detected events

        window (int): maximum distance for the correlation between fact and detection

        return_match (bool): returns the matching tuple idx [(fact_idx, detection_idx),...] if set true



    Returns:

        dict: {'tp':int, 'fp':int, 'fn':int, 'precision':float, 'recall':float, 'dis':float, 'match': list of tuple}

    """

    if len(fact) == 0 or len(detection) == 0:

        return evaluation_window(fact, detection, window, return_match)



    cost_matrix = make_cost_matrix(fact, detection, window)

    # handle the case there is actually no edges between fact and detection

    if all([cost_matrix[i][j] == sys.maxsize for i in range(len(fact)) for j in range(len(detection))]):

        summary = dict(tp=0, fp=len(detection), fn=len(fact),

                       precision=0, recall=0,

                       dis=None, match=[])

        return summary



    cut = cut_matrix(cost_matrix, sys.maxsize)  # [((fact/line range), (detect/column range)),...]

    match_cut = [evaluation_window(fact[i[0][0]:i[0][1]], detection[i[1][0]:i[1][1]], window, True) for i in cut]



    tp = sum([i['tp'] for i in match_cut if i['tp']])  # in general is not possible to have i['tp'] is None

    fp = len(detection) - tp

    fn = len(fact) - tp



    match = []

    for i, res in enumerate(match_cut):

        match.extend([(f+cut[i][0][0], d+cut[i][1][0]) for f, d in res['match']])  # adjust index according to starting



    summary = dict(tp=tp, fp=fp, fn=fn,

                   precision=float(tp) / (tp + fp) if len(detection) > 0 else None,

                   recall=float(tp) / (tp + fn) if len(fact) > 0 else None,

                   dis=sum([abs(fact[i]-detection[j]) for i, j in match]) / float(tp) if tp > 0 else None)



    if return_match:

        summary['match'] = match



    return summary





def cut_matrix(mat, no_edge=0):

    """ given a cost matrix, cut it into non-connecting parts



    For example:

    cost_matrix = [[62, 0,  0,  0, 0,  0, 0],

                   [11, 11, 82, 0, 0,  0, 0],

                   [0,  0, 81, 12, 0,  0, 0],

                   [0,  0,  0,  0, 12, 0, 0],

                   [0,  0,  0,  0, 0,  0, 0],

                   [0,  0,  0,  0, 0,  0, 12],

                   [0,  0,  0,  0, 0,  0, 12]]

    expect return: [((0, 4), (0, 5)), ((3, 4), (4, 5)), ((5,7),(6,7))]

    Input like this is as well acceptable, though such case is not possible in the usage of this project.

    cost_matrix = [[62, 0,  0,  0, 0,  0, 0],

                   [11, 11, 82, 0, 0,  0, 0],

                   [0,  0, 81, 12, 0,  0, 0],

                   [0,  0, 12,  0, 0,  0, 0],

                   [0,  0,  0,  0, 0,  0, 12],

                   [0,  0,  0,  0, 0, 11, 12],

                   [0,  0,  0,  0, 0,  0, 12]]

    the lower-righter sub-matrix doesn't have edge as the top left corner.



    Args:

        mat (list of list of equal length): the cost matrix

        no_edge (int): the value in matrix meaning the the two nodes are not connected, thus no_edge



    Return:

        list of tuple: [((row from, to), (column from, to)), (another sub-matrix)...]

    """

    def cutter(mat, righter, downer):

        """ given the matrix and the two outer surrounding coordinates of the top left corner of a submatrix



        righter and downer traces the outer contour of a submatrix and verifies where it ends.

        righter goes right (increment in column index) when the value to its right is not an edge, else goes downwards.

        downer goes downside (increment in row index) when the value beneath it is not an edge, else goes right.

        righter and downer cuts a sub-matrix if they are in a diagonal position, corner touch corner.



        Args:

            mat (list of list of equal length): the cost matrix

            righter (tuple of two int): coordinate of righter

            downer (tuple of two int): coordinate of downer



        Returns:

            cut (tuple of two int): the row and column index the cuts (outer border) the sub-matrix beginning from the point

            surrounded by the input righter and downer

        """

        righter_copy = righter  # save the initial righter

        righter_set = set()  # the righter position ever visited

        cut = (len(mat), len(mat[0]))  # the default return value, if not cut, righter downer matches there, outside matrix

        # trace the righter first, to the very end

        # the stop condition is when the column index reaches the column number of the matrix + 1

        while righter[1] <= len(mat[0]):

            righter_set.add(righter)

            # righter can move right if it is already out side the matrix or next value is not an edge

            if righter[0] == len(mat) or (righter[1]+1 < len(mat[0]) and mat[righter[0]][righter[1]+1] == no_edge):

                righter = (righter[0], righter[1]+1)

            else:

                # otherwise move downwards

                righter = (righter[0]+1, righter[1])

        righter_set.remove(righter_copy)  # remove the initial righter so that it won't match up with first downer

        # then move the downer, the stop condition its row index is matrix row number + 1

        while downer[0] <= len(mat):

            # in general initial downer always matches with the initial righter, why removing the initial righter

            if (downer[0]+1, downer[1]-1) in righter_set:  # test diagonal position

                cut = (downer[0]+1, downer[1])

                break

            # can move down if already outside matrix or next value is not an edge

            if downer[1] == len(mat[0]) or (downer[0] + 1 < len(mat) and mat[downer[0]+1][downer[1]] == no_edge):

                downer = (downer[0]+1, downer[1])

            else:  # other wise move right

                downer = (downer[0], downer[1]+1)

        # if not cut righter surely contain (len(mat), len(mat[0]-1))

        # then downer matches righter at (len(mat)-1, len(mat[0]))

        # which makes the default cut that is (len(mat), len(mat[0]))

        return cut



    # the crossing point (inclusive) of line_start and column start is the top left corner of sub-matrix

    line_start = 0

    column_start = 0

    res = []  # row and column index range for each submatrix

    while line_start < len(mat) and column_start < len(mat[0]):



        righter = None

        downer = None

        for i in range(line_start, len(mat)):

            # the righter is the position left to the top element in the first non-empty column

            row = mat[i]

            if any([v != no_edge for v in row]):

                downer = (i-1, [j for j, v in enumerate(row) if v != no_edge][0])

                break



        for i in range(column_start, len(mat[0])):

            # the downer is the position upper to the first element in the first non-empty row

            column = [row[i] for row in mat]

            if any([v != no_edge for v in column]):



                righter = ([j for j, v in enumerate(column) if v != no_edge][0], i-1)

                break

        # if can not be found means from line_start, column_start, there is no edge left

        if righter is None or downer is None:

            break



        line_start, column_start = cutter(mat, righter, downer)  # update starting point with the last cut

        # ((row index range), (column index range))

        res.append(((min(righter[0], downer[0]+1), line_start), (min(righter[1]+1, downer[1]), column_start)))



    return res

def evaluation_window(fact, detection, window=0, return_match=False):
    """classify the detections with window option
    We construct a bipartite graph G = (V + W, E), where V is fact and W is detection.
    e = (v, w), e in G, if distance(v, w) <= window.
    cost(e) = distance(v, w)
    We find the minimum-cost maximum matching M of G.
    tp = |M|
    fp = |W| - |M|
    fn = |V| - |M|
    dis = C(M)/|M| average distance between fact and detection in mapping
    Args:
        fact (list of int): the index or timestamp of facts/events to be detected
        detection (list of int): index or timestamp of detected events
        window (int): maximum distance for the correlation between fact and detection
        return_match (bool): returns the matching tuple idx [(fact_idx, detection_idx),...] if set true
    Returns:
        dict: {'tp':int, 'fp':int, 'fn':int, 'precision':float, 'recall':float, 'dis':float, 'match': list of tuple}
    """
    if len(fact) == 0:
        summary = dict(tp=None, fp=len(detection), fn=None,
                       precision=None, recall=None,
                       dis=None, match=[])
        return summary
    elif len(detection) == 0:
        summary = dict(tp=0, fp=0, fn=len(fact),
                       precision=None, recall=0,
                       dis=None, match=[])
        return summary

    cost_matrix = make_cost_matrix(fact, detection, window)  # construct the cost matrix of bipartite graph

    # handle the case there is actually no edges between fact and detection
    if all([cost_matrix[i][j] == sys.maxsize for i in range(len(fact)) for j in range(len(detection))]):
        summary = dict(tp=0, fp=len(detection), fn=len(fact),
                       precision=0, recall=0,
                       dis=None, match=[])
        return summary

    match = munkres.Munkres().compute(cost_matrix)  # calculate the matching
    match = [(i, j) for i, j in match if cost_matrix[i][j] <= window]  # remove dummy edges
    # i and j here are the indices of fact and detection, i.e. ist value in fact and jst value in detection matches

    tp = len(match)
    fp = len(detection) - tp
    fn = len(fact) - tp

    summary = dict(tp=tp, fp=fp, fn=fn,
                   precision=float(tp) / (tp + fp) if len(detection) > 0 else None,
                   recall=float(tp) / (tp + fn) if len(fact) > 0 else None,
                   dis=sum([cost_matrix[i][j] for i, j in match]) / float(tp) if tp > 0 else None)

    if return_match:
        summary['match'] = match

    return summary

def make_cost_matrix(x, y, window):

    """ make cost matrix for bipartite graph x, y"""

    return [[abs(x[i] - y[j]) if abs(x[i]-y[j]) <= window else sys.maxsize for j in range(len(y))] for i in range(len(x))]

#y_pred_statusChange=np.zeros(y_pred.shape[0])
#y_test_statusChange=np.zeros(y_test.shape[0])
y_pred_statusChange=[]
y_test_statusChange=[]

for i in range(y_pred.shape[0]-1):
    if y_pred[i]!=y_pred[i+1]:
        #y_pred_statusChange[i+1]=1
        y_pred_statusChange.append(i+1)
    if y_test[i]!=y_test[i+1]:
        #y_test_statusChange[i+1]=1
        y_test_statusChange.append(i+1)
        
evaluation_window_adp(y_test_statusChange,y_pred_statusChange,window=2)

t1=np.arange(0,y_test.shape[0]*2,2)
t2=np.arange(0,y_pred.shape[0]*2,2)

plt.plot(t1[7500:9000],y_test[7500:9000])

plt.plot(t2[7500:9000],y_pred[7500:9000])

plt.xlabel("Time (s)")
plt.ylabel("Label")
plt.show()