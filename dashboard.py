import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server
scaler=MinMaxScaler(feature_range=(0,1))


df_BTC = pd.read_csv("./BTC-USD.csv")
df_ETH = pd.read_csv("./ETH-USD.csv")
df_ADA = pd.read_csv('./ADA-USD.csv')

def prediction(df, modelSrc):
    df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index=df['Date']
    data=df.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])


    for i in range(0,len(data)):
        new_data['Date'][i]=data['Date'][i]
        new_data['Close'][i]=data['Close'][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)
    dataset=new_data.values
    train=dataset[0:275,:]
    valid=dataset[275:,:]
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)
    x_train,y_train=[],[]


    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
        
    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    model=load_model(modelSrc)

    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)
    X_test=[]

    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])

    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)

    train=new_data[:275]
    valid=new_data[275:]
    valid['Predictions']=closing_price

    return train, valid

train_BTC, valid_BTC = prediction(df_BTC, "./lstm_btc.h5")
train_ETH, valid_ETH = prediction(df_ETH, "./lstm_eth.h5")
train_ADA, valid_ADA = prediction(df_ADA, "./lstm_ada.h5")

df = pd.concat([pd.read_csv("./coin_Bitcoin.csv"), pd.read_csv("./coin_Ethereum.csv"), pd.read_csv("./coin_Cardano.csv")],ignore_index=True)

df['SNo'] = df.index

df.head()

app.layout = html.Div([
   
    html.H1("Crypto Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Crypto price',children=[
            html.Div([
                dcc.Dropdown(id='my-dropdown1',
                             options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Etherium','value': 'ETH'},
                                      {'label': 'Cardano', 'value': 'ADA'}], 
                             multi=False,value='BTC',
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_BTC.index,
                                y=valid_BTC["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_BTC.index,
                                y=valid_BTC["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )                
            ])                
        ]),
        dcc.Tab(label='Crypto Data', children=[
            html.Div([
                html.H1("BTC_USD High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Etherium','value': 'ETH'},
                                      {'label': 'Cardano','value': 'ADA'}], 
                             multi=True,value=['BTC'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown3',
                         options=[{'label': 'Bitcoin', 'value': 'BTC'},
                                      {'label': 'Etherium','value': 'ETH'},
                                       {'label': 'Cardano','value': 'ADA'}], 
                             multi=True,value=['BTC'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])
@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"BTC": "Bitcoin","ETH": "Etherium","ADA": "Cardano"}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure
@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown3', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"BTC": "Bitcoin","ETH": "Etherium","ADA": "Cardano"}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure

@app.callback(Output('Actual Data', 'figure'),
              Input('my-dropdown1', 'value'))
def update_graph(selected_dropdown_value):
    valid = []
    train = []
    match selected_dropdown_value:
        case "BTC":
            valid = valid_BTC
            train = train_BTC
        case "ETH":
            valid = valid_ETH
            train = train_ETH
        case "ADA":
            valid = valid_ADA
            train = train_ADA
        case _:
            valid = []

    figure={
                "data":[
                    go.Scatter(
                        x=train.index,
                        y=valid["Close"],
                        mode='markers'
                    )
                ],
                "layout":go.Layout(
                    title='scatter plot',
                    xaxis={'title':'Date'},
                    yaxis={'title':'Closing Rate'}
                )
            }
    return figure

@app.callback(Output('Predicted Data', 'figure'),
              Input('my-dropdown1', 'value'))
def update_graph(selected_dropdown_value):
    valid = []
    match selected_dropdown_value:
        case "BTC":
            valid = valid_BTC
        case "ETH":
            valid = valid_ETH
        case "ADA":
            valid = valid_ADA
        case _:
            valid = []

    figure={ 
               "data":[
                    go.Scatter(
                        x=valid.index,
                        y=valid["Predictions"],
                        mode='markers'
                    )
                    ],
                    "layout":go.Layout(
                        title='scatter plot',
                        xaxis={'title':'Date'},
                        yaxis={'title':'Closing Rate'}
                    )
            }
    return figure


if __name__=='__main__':
    app.run_server(debug=True)