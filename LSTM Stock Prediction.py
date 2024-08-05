import pandas as pd
import requests
import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error

# 1. GETTING AND DOWNLOADING THE DATA

data_source = "alphavantage"
api_key = "FPR74LJ6B7U3Y6RK" 
ticker = "AAPL" #Listed Stock Name 
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key) # JSON file with all the stock market data for JP Morgan from the last 20 years
file_to_saveas = 'stock_market_data-%s.csv'%ticker

if not os.path.exists(file_to_saveas): # If file does not exist
     try:
         response = requests.get(url_string) # Gets the data from URL as a request.Response object
         response.raise_for_status() # Raises errors for HTTP issues
         json_data = response.json() # Converts Response object to readable JSON
         json_data = json_data['Time Series (Daily)'] # Reassigns json_data to the dictionary containing only the time series data, eliminating metadata
         df = pd.DataFrame(columns = ['Date', 'Low', 'High', 'Close', 'Open']) # Constructor for pandas dataframe
         for k, v in json_data.items(): # k(key) is the date, v(value) is the dictionary containing the high, low, open and close prices
             date = dt.datetime.strptime(k, '%Y-%m-%d') # Converting date in data from string to datetime object
             data_row = [date.date(), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['1. open'])] # List of elements to add to the dataframe
             df.loc[len(df)] = data_row # Appending list of elements to dataframe
         df.set_index('Date', inplace=True) # Sets the index of the dataframe as the Data Column
         df.sort_index(inplace=True) # Sorts the dataframe in ascending order with respect to the date column
         df.to_csv(file_to_saveas) # Saving pandas dataframe as CSV
         print("Data saved to filename : %s"%(file_to_saveas))
     except Exception as e: print(e)
else:
     print('File already exists. Loading data from CSV')
     df = pd.read_csv(file_to_saveas) # Making pandas dataframe from CSV

print("Total no. of data points downloaded : ", df.shape[0])

# 2. DATA PREPROCESSING

# VISUALIZATION
plt.figure(figsize = (18, 9)) # sets the size of the figure to be 18 inches wide and 9 inches tall 
plt.plot(range(df.shape[0]), (df['Close'])) #plots graph of parameters x(df.shape[0] = total no. of rows i.e. values) and y(average price of that day)
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Time Points', fontsize = 15)
plt.ylabel('Closing Price / $', fontsize = 15)
plt.title("%s Stock Data"%(ticker), fontsize = 18, pad = 20)
plt.show()

closing_prices = df.loc[:, 'Close'].to_numpy().reshape(-1, 1) # Use to_numpy().reshape(-1, 1) if you need a 2D array for operations that require specific input shapes, such as in scikit-learn where many functions expect 2D arrays for features.
training_size = int(len(closing_prices) * 0.60)
train_data = closing_prices[:training_size].reshape(-1, 1) # First <training_size> values are training data, (x,) to (x, 1)
test_data = closing_prices[training_size:].reshape(-1, 1) # Remaining values are test data

scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

all_normalized_data = np.concatenate([train_data, test_data], axis=0)

'''
def rolling_window_normalization(window_size):
    for di in range(0,len(train_data), window_size):
        end = min(di + window_size, len(train_data)) # to avoid empty array error
        scaler.fit(train_data[di:end]) # training the scaler
        data_max[di] = max(train_data[di:end])
        data_min[di] = min(train_data[di:end])
        train_data[di:end] = scaler.transform(train_data[di:end])
'''

plt.figure(figsize=(18,9))
plt.plot(range(df.shape[0]), all_normalized_data)
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45) # parameters = plt.xticks(how many ticks, what to label those ticks, rotation of labels)
plt.xlabel('Time Points', fontsize = 18)
plt.ylabel('Normalized Values', fontsize = 18)
plt.title("%s Stock Data"%(ticker), fontsize = 18, pad = 20)
plt.show()


# 3. GENERATING DATASET

'''
VARIABLE EXPLANATION
xtrain - collection of sequences consisting of <timesteps> no. of training data
ytrain - the training data immediately following the sequence with the corresponding index (actual output to train the model)
xtest - collection of sequences consisting of <timesteps> no. of testing data
ytest - the testing data immediately following the sequence with the corresponding index (actual output to test the model)
'''

def CreateDataMatrix(dataset, timestep=2): # One Day Ahead
     independant_data, dependant_data = [], []
     for i in range(len(dataset)-timestep-1):
         data_row = dataset[i:i+timestep, 0]
         independant_data.append(data_row)
         dependant_data.append(dataset[i+timestep, 0])
     return np.array(independant_data), np.array(dependant_data)

timestep = len(closing_prices) // 40
xtrain, ytrain = CreateDataMatrix(train_data, timestep)
xtest, ytest = CreateDataMatrix(test_data, timestep)

# 4. CREATING A STACKED LSTM MODEL
'''
Before creating an LSTM Model, the Input Arrays MUST be reshaped to be 3D [samples, timesteps, features] which is required for an LSTM.
'''

xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)

print(xtrain.shape, ytrain.shape) # Test
print(xtest.shape, ytest.shape) # Test

tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape = (timestep, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(xtrain, ytrain, validation_data = (xtest, ytest), epochs=50, batch_size = 64, verbose = 1, callbacks=[early_stopping]) # Training the LSTM Model

# VISUALIZING THE TRAINING LOSS AGAINST VALIDATION LOSS

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(18, 9))
plt.plot(loss, color='purple', label='Training Loss')
plt.plot(val_loss,  color = 'green', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


train_predict = model.predict(xtrain)
test_predict = model.predict(xtest)

ytrain = ytrain.reshape(ytrain.shape[0], 1)
ytest = ytest.reshape(ytest.shape[0], 1)

print(math.sqrt(mean_squared_error(ytrain, train_predict))) # RMSE Performance Metrics over Training Data
print(math.sqrt(mean_squared_error(ytest, test_predict))) # RMSE Performance Metrics over Testing Data

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

train_predict_plot = np.empty_like(closing_prices) # Initializing an empty array of the same shape as <closing prices>
train_predict_plot[:, :] = np.nan # Filling all elements in the array with NaN i.e. Not a Number which helps in visualizing gaps where predictions do not exist
train_predict_plot[timestep:len(train_predict) + timestep, :] = train_predict #

test_predict_plot = np.empty_like(closing_prices) # Initializing an empty array of the same shape as <closing prices>
test_predict_plot[:, :] = np.nan # Filling all elements in the array with NaN i.e. Not a Number which helps in visualizing gaps where predictions do not exist
test_predict_plot[len(train_predict) + (timestep * 2) + 1:len(closing_prices) - 1, :] = test_predict



plt.figure(figsize=(18,9))
plt.plot(df.index, closing_prices, label='Actual')
plt.plot(df.index, train_predict_plot, label='Train Prediction')
plt.plot(df.index, test_predict_plot, label='Test Prediction')
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Closing Price / $', fontsize = 18)
plt.title(ticker, " Price Data")
plt.legend()
plt.show()

'''
def CreateDataMatrix_XDA(dataset, num_of_days=30, timestep=2): # XDaysAhead
     independant_data, dependant_data = [], []
     for i in range(len(dataset)-timestep-num_of_days):
         independant_data_row = dataset[i:i+timestep, 0]
         independant_data.append(data_row)
         dependant_data_row = dataset[]
         dependant_data.append(dataset[i+timestep:i+timestep+num_of_days+1, 0])
     return np.array(independant_data,), np.array(dependant_data)
'''