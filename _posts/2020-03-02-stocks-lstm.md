---
title: 'Stock Price Predictions with LSTM.'
date: 2020-03-02
permalink: /posts/2019/03/stocks-lstm
tags:
  - Finance
  - Python
  - Artificial Intelligence
---

Import all the required libraries:
```python
import cufflinks as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import plotly
import plotly.offline as plyo
import yfinance as yf
yf.pdr_override()

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

```
We will use pandas datareader and yfinance to fetch ticker data from yahoo fiance. Let's create a function which will accept three arguments, stock symbol, start date, and end date.
```python
#  SYMBOL  -- the symbol of the stock
#  start -- the starting date of the time period
#  end   -- end date of the time period

def get_ticker_data(SYMBOL, start, end):
    
    stock_data = pdr.get_data_yahoo(SYMBOL, start, end)
    return stock_data

```
Get the daily ticker data of Apple Inc. from January, 2007 till August, 2019.
```python
df = get_ticker_data("AAPL", "2007-01-01", "2019-08-30")
df.head()
```
![png](/images/plots/stocklstm1.png )

```python
df.tail()
```
![png](/images/plots/stocklstm2.png )

Let's check the shape of our dataframe.
```python
df.shape[0]

Output:
3187
```
Now, we will create the Open & Close price chart for last 60 days.
```python
df_oc = get_ticker_data("AAPL", "2019-01-01", "2019-06-30")
df_plot = pd.DataFrame(df_oc, columns=['Open', 'Close'])
plyo.plot(df_plot.iplot(asFigure=True), filename="daily.html",auto_open=False)

```
<iframe width="100%" height="500" src="/images/plots/daily.html">stocks</iframe>
<br />

And, here is the candle chart for last 60 days.
```python
df_candle = df.copy()
quotes = df_candle[['Open', 'High', 'Low', 'Close']]
quotes = quotes.iloc[-60:]
qf = cf.QuantFig(quotes, title='Apple Inc.', legend='top', name='AAPL')
plyo.plot(qf.iplot(asFigure=True), filename="exchange_rate.html",auto_open=False)

```
<iframe width="100%" height="500" src="/images/plots/candle.html">stocks</iframe>
<br />

Create training and test set.
```python
training_set = df[:'2017'].iloc[:,1:2].values
test_set = df['2018':].iloc[:,1:2].values
df["High"][:'2017'].plot(figsize=(16,5),legend=True, color="red")
df["High"]['2018':].plot(figsize=(16,5),legend=True, color="black")
plt.legend(['Training set','Test set'])
plt.title('APPLE stock price')
plt.show()

```
![png](/images/plots/stocklstm3.png )

Create the LSTM model with dropout regularization.
```python

# Scaling the training set
scaled_data = MinMaxScaler(feature_range=(0,1))
scaled_training_set = scaled_data.fit_transform(training_set)


#100 sequences at once
X_train = []
y_train = []
for i in range(100,len(scaled_training_set)):
    X_train.append(scaled_training_set[i-100:i,0])
    y_train.append(scaled_training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping for LSTM
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# The LSTM Model
model = Sequential()

# First layer.
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
#Dropout regularisation
model.add(Dropout(0.2))

# Second layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Third layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Fourth layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='rmsprop',loss='mean_squared_error')

# Fitting to the training set
model.fit(X_train,y_train,epochs=5,batch_size=32)
```

```python
Output:
Epoch 1/5
2669/2669 [==============================] - 70s 26ms/step - loss: 0.0141
Epoch 2/5
2669/2669 [==============================] - 66s 25ms/step - loss: 0.0062
Epoch 3/5
2669/2669 [==============================] - 67s 25ms/step - loss: 0.0051
Epoch 4/5
2669/2669 [==============================] - 67s 25ms/step - loss: 0.0046
Epoch 5/5
2669/2669 [==============================] - 66s 25ms/step - loss: 0.0036

<keras.callbacks.History at 0x7fbdf492c400>
```
Create a new dataframe by concatenating the values in the 'High' column.
```python
df_high = pd.concat((df["High"][:'2017'],df["High"]['2018':]),axis=0)

```
Generate predictions.
```python
inputs = df_high[len(df_high)-len(test_set) - 100:].values
inputs = inputs.reshape(-1,1)
inputs  = scaled_data.transform(inputs)
X_test = []
for i in range(100,len(inputs)):
    X_test.append(inputs[i-100:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaled_data.inverse_transform(predicted_stock_price)
```

Create the real price vs predicted price graph.
```python
df_pred = pd.DataFrame(test_set, columns=['real'])
df_pred['predicted']=predicted_stock_price
plyo.plot(df_pred.iplot(asFigure=True), filename="pred.html",auto_open=False)

```
<iframe width="100%" height="500" src="/images/plots/pred.html">stocks</iframe>
