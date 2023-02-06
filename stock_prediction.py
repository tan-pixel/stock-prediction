#import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import plotly.express as px

step = 500
start = '2000-01-01'
end = '2021-11-11'

st.title('Stock Prediction')

input_name = st.text_input('Enter Stock', 'AAPL')
df = web.DataReader(input_name, 'yahoo', start, end)
df = df.reset_index()

#st.subheader('Data')
st.write(df.describe())

#%%
def plot_raw():
  #fig = go.Figure()
  fig = px.line(x = df.Date, y = df.Close, width=1000, height=600)
  fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='Close'
  st.plotly_chart(fig)

st.subheader(f'Close price data for {input_name}')
plot_raw()

indicators = pd.DataFrame()

#%%
def ma(df, period = 30):
  return df.Close.rolling(period).mean()

indicators['MA'] = ma(df)
def plot_ma():
  fig = px.line(x = df.Date, y = [df.Close, indicators['MA']], width=1000, height=600)
  fig.layout.update(xaxis_rangeslider_visible = True)
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='Close'
  fig['data'][1]['showlegend']=True
  fig['data'][1]['name']='Simple moving average'
  st.plotly_chart(fig)

st.subheader('Simple moving average')
plot_ma()

#%%
def wma(df, period = 30):
  common_diff = 2/(period*(period+1))
  weights = np.linspace(common_diff, period*common_diff, period)
  return df.Close.rolling(period).apply(lambda x: np.sum(weights*x))

indicators['WMA'] = wma(df)
def plot_wma():
  fig = px.line(x = df.Date, y = [indicators['MA'], indicators['WMA']], width=1000, height=600)
  fig.layout.update(xaxis_rangeslider_visible = True)
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='Simple moving average'
  fig['data'][1]['showlegend']=True
  fig['data'][1]['name']='Weighted moving average'
  st.plotly_chart(fig)

st.subheader('Weighted moving average')
plot_wma()

#%%
def ema(df, period = 30):
  return df.Close.ewm(period, adjust = False).mean()

indicators['EMA'] = ema(df)
def plot_ema():
  fig = px.line(x = df.Date, y = [indicators['MA'], indicators['EMA']], width=1000, height=600)
  fig.layout.update(xaxis_rangeslider_visible = True)
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='Simple moving average'
  fig['data'][1]['showlegend']=True
  fig['data'][1]['name']='Exponential moving average'
  st.plotly_chart(fig)

st.subheader('Exponential moving average')
plot_ema()

#%%
def MACD(df, long_prd = 26, short_prd = 12, signal_prd = 9):
  return ema(df, short_prd) - ema(df, long_prd), ema(df, signal_prd)

indicators['MACD'], indicators['signal'] = MACD(df)
def plot_macd():
  fig = px.line(x = df.Date, y = [indicators['MACD'], indicators['signal']], width=1000, height=600)
  fig.layout.update(xaxis_rangeslider_visible = True)
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='MACD'
  fig['data'][1]['showlegend']=True
  fig['data'][1]['name']='Signal line'
  st.plotly_chart(fig)

st.subheader('Moving Average Convergence Divergence')
plot_macd()

#%%
def rsi(df, period = 30):
  delta = df.Close.diff(1)[1:]
  up, down = delta.copy(), delta.copy()
  up[up<0], down[down>0] = 0, 0
  avg_gain = up.rolling(period).mean()
  avg_loss = abs(down.rolling(period).mean())
  rs = avg_gain/avg_loss
  rsi_ = 100.0 - 100.0/(1+rs)
  return rsi_

indicators['RSI'] = rsi(df)
def plot_rsi():
  fig = px.line(x = df.Date, y = [indicators['RSI']], width=1000, height=600)
  fig.layout.update(xaxis_rangeslider_visible = True)
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='RSI'
  st.plotly_chart(fig)

st.subheader('Relative Strength Index')
plot_rsi()

#%%

model = load_model('model_2.h5')
data = pd.DataFrame(df.Close)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

y_test = np.array([])
flag = 0

for i in range(step, scaled_data.shape[0]):
  if flag == 0:
    X_test = np.array(scaled_data[i-step:i, 0]) 
    flag = 1
  else:
    X_test = np.vstack((X_test, scaled_data[i-step:i, 0])) 
  y_test = np.append(y_test, scaled_data[i])

X_future = X_test[-1].ravel()
y_future = np.array([])
prd = 30
for i in range(prd):
  y_future_pred = model.predict(X_future.reshape(1, -1))
  X_future = np.append(X_future[1:], y_future_pred)
  y_future = np.append(y_future, y_future_pred)

y_pred = model.predict(X_test)

#%%
idx = np.arange(len(y_pred))
idx_fut = np.arange(len(y_pred)-1, len(y_pred) + prd - 1)
y_pred /= scaler.scale_
y_test /= (scaler.scale_)
y_future /= (scaler.scale_)

y_pred = np.reshape(y_pred, (-1,))

def plot_prediction():
  fig = px.line(x = idx, y = [y_test, y_pred], width=1000, height=600)
  fig2 = px.line(x = idx_fut, y = y_future, width=1000, height=600)
  fig2.update_traces(line_color='green')
  fig['data'][0]['showlegend']=True
  fig['data'][0]['name']='Close'
  fig['data'][1]['showlegend']=True
  fig['data'][1]['name']='Prediction'
  fig2['data'][0]['showlegend']=True
  fig2['data'][0]['name']='Future prediction'
  fig.add_trace(fig2.data[0])
  fig.layout.update(xaxis_rangeslider_visible = True)
  st.plotly_chart(fig)

st.subheader('LSTM prediction')
plot_prediction()
