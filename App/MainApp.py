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

app = dash.Dash('vehicle-data')


max_length = 50
times = deque(maxlen=max_length)
f1 = deque(maxlen=max_length)
f2 = deque(maxlen=max_length)
f3 = deque(maxlen=max_length)
f4 = deque(maxlen=max_length)

data_dict = {"F1(Count)":f1,
"F2(srv_count)": f2,
"F3(dst_host_count)": f3,
"F4(dst_host_srv_count)":f4
}



def update_obd_values(times, f1, f2, f3, f4):

    times.append(time.time())
    if len(times) == 1:
        #starting relevant values
        f1.append(random.randrange(180,230))
        f2.append(random.randrange(95,115))
        f3.append(random.randrange(170,220))
        f4.append(random.randrange(1000,9500))
    else:
        for data_of_interest in [f1, f2, f3, f4]:
            data_of_interest.append(data_of_interest[-1]+data_of_interest[-1]*random.uniform(-0.0001,0.0001))

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
            name='Scatter',
            fill="tozeroy",
            fillcolor="#6897bb"
            )

        graphs.append(html.Div(dcc.Graph(
            id=data_name,
            animate=True,
            figure={'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(times),max(times)]),
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