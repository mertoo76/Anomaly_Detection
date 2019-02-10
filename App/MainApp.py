# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:42:23 2019

@author: user
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader.data import DataReader
import time
from collections import deque
import plotly.graph_objs as go
import random

import pandas as pd
import numpy as np


app = dash.Dash('vehicle-data')


import subprocess
#subprocess.check_call(['Rscript', 'EMD.R'], shell=False)
# Define command and arguments
command = 'Rscript'
path2script = 'EMD.R'


x0=[]
y0=[]

#importing Neural Network
import keras
def load_Model():
    from keras.models import model_from_yaml
    yaml_file = open('LSTM/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("LSTM/model.h5")
    print("Loaded model from disk")
    return loaded_model

#Load Model
model=load_Model()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#importing the dataset
datasetTest = pd.read_csv('../New/KDD10Percentage/Data/corrected',',')

x_test=datasetTest.iloc[:,22:24]
x_test=np.append(x_test, datasetTest.iloc[:,31:33], axis=1)
y_test = datasetTest.iloc[:, 41].values

x_test=x_test[800:]
y_test=y_test[800:]

for i in range(len(y_test)):
    if y_test[i] != 'normal.':
        #print(y_prep[i])
        y_test[i]='attack'
    
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)

max_length = 50
times = deque(maxlen=max_length) #time information
f1 = deque(maxlen=max_length) #Feature1
f2 = deque(maxlen=max_length) #Feature2 
f3 = deque(maxlen=max_length) #Feature3
f4 = deque(maxlen=max_length) #Feature4

#times.append(0)

data_dict = {"F1(Count)":f1,
"F2(srv_count)": f2,
"F3(dst_host_count)": f3,
"F4(dst_host_srv_count)":f4
}



def update_obd_values(times, f1, f2, f3, f4):

    #times.append(time.time())
    if len(times)==0:
        times.append(0)
        f1.append(x_test[times[-1]][0])
        f2.append(x_test[times[-1]][1])
        f3.append(x_test[times[-1]][2])
        f4.append(x_test[times[-1]][3])
        
    else:
        times.append(times[-1]+1)
        f1.append(x_test[times[-1]][0])
        f2.append(x_test[times[-1]][1])
        f3.append(x_test[times[-1]][2])
        f4.append(x_test[times[-1]][3])
             
    
    
    return times, f1, f2, f3, f4

times, f1, f2, f3, f4 = update_obd_values(times, f1, f2, f3, f4)

app.layout = html.Div([
    html.Div([
        html.H2('Anomaly Detection',
                style={'float': 'left',
                       }),
        ]),
    dcc.Dropdown(id='vehicle-data-name',
                 options=[{'label': s, 'value': s}
                          for s in data_dict.keys()],
                 value=['F1(Count)','F2(srv_count)','F3(dst_host_count)'],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),
    dcc.Interval(
        id='graph-update',
        interval=1*1000,
        n_intervals=0),
    ], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})


@app.callback(
    dash.dependencies.Output('graphs','children'),
    [dash.dependencies.Input('vehicle-data-name', 'value'),
     dash.dependencies.Input('graph-update', 'n_intervals')]
    )
def update_graph(data_names,Interval):
    global x0,y0,command,path2script,model
    
    graphs = []
    update_obd_values(times, f1, f2, f3, f4)
    if len(data_names)>2:
        class_choice = 'col s12 m6 l4'
    elif len(data_names) == 2:
        class_choice = 'col s12 m6 l6'
    else:
        class_choice = 'col s12'


    for data_name in data_names:

        data = go.Scatter(
            x=list(times),
            y=list(data_dict[data_name]),
            name='data',
            fill="tozeroy",
            fillcolor="#6897bb"
            )
        
        
        #Attack
        if times[-1] % 10 == 0:
            print("------------------------attack--------------------------")
            x_test_tmp = np.zeros((10,4)) 
            x_test_tmp[:,0] = list(reversed([f1[-i] for i in range(1,11)]))
            x_test_tmp[:,1] = list(reversed([f2[-i] for i in range(1,11)]))
            x_test_tmp[:,2] = list(reversed([f3[-i] for i in range(1,11)]))
            x_test_tmp[:,3] = list(reversed([f4[-i] for i in range(1,11)]))
            #print("Python x_test_tmp: ", x_test_tmp)
            #call EMD here
            # Variable number of args in a list
            args = np.zeros(40)
            for i in range(x_test_tmp.shape[1]):
                args[10*i:10*(i+1)]=x_test_tmp[:,i]
            # Build subprocess command
            cmd = [command, path2script] + [str(i) for i in args]
            # check_output will run the command and store to result
            #x = subprocess.check_output(cmd, universal_newlines=True)
            #print("Python args: ",args)
            subprocess.call(cmd, universal_newlines=True,shell=True)
            
            #Get EMD info
            emd = pd.read_csv('emd.csv',',')
            emd=emd.values
            #print("EMD Values: Leng:",emd.shape[0]," ",emd.shape[1]," ",emd,"\n")

            emd = emd.reshape((1,10,emd.shape[1]))
            # Predicting the Test set results
            y_pred = model.predict(emd)
            y_pred = (y_pred > 0.5)
            y_pred=y_pred.reshape((y_pred.shape[0]*y_pred.shape[1]))
            print("Prediction: ", y_pred)
        
        
            y_pred=y_pred*1
            for i,item in enumerate(times):
                newIt=np.asarray(item)%10
                if y_pred[newIt.tolist()] == 0:
                    x0.append(item)
                    y0.append(data_dict[data_name][i])            

             
        """    
        x0=[]
        y0=[]
        for i,item in enumerate(times):
            if y_test[item] == 0:
                x0.append(item)
                y0.append(data_dict[data_name][i])
        """     
        
        index=[i for i in range(len(x0)) if x0[i] in list(times) ]
        x0=list(np.asarray(x0)[index])
        y0=list(np.asarray(y0)[index])
        #print("Y: ",y0,"\n")
        attack = go.Scatter(
                x=x0,
                y=y0,
                mode='markers',
                name='attack',
                marker = dict(
                        size = 10,
                        color = 'rgba(152, 0, 0, .8)'
                       )         
            )

        graphs.append(html.Div(dcc.Graph(
            id=data_name,
            animate=True,
            figure={'data': [data,attack],'layout' : go.Layout(xaxis=dict(range=[min(times),max(times)]),
                                                        yaxis=dict(range=[min(data_dict[data_name]),max(data_dict[data_name])]),
                                                        margin={'l':50,'r':1,'t':45,'b':1},
                                                        title='{}'.format(data_name))}
            ), className=class_choice))

    return graphs



external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})


if __name__ == '__main__':
    app.run_server(debug=True)