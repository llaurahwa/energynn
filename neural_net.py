#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:11:05 2022

@author: laurah
"""

#import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#import data file
data = pd.read_csv(r'/Users/laurah/Desktop/code/neural net proj/train.csv')

def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

#delete meter readings of 0
condition = data["meter_reading"]>1e-10
data_clean = data[condition]
column_names = ["timestamp","meter_reading"]

#examine time and meter reading
data_clean = data_clean[["timestamp", "meter_reading"]]
data_clean = data_clean.astype({"timestamp": str})
data_clean = data_clean.astype({"meter_reading": float})
data_clean.reset_index(drop = True, inplace = True)

#create new pandas array to populate
newdata = pd.DataFrame(index = range(len(data_clean)), columns = column_names)

#create new pandas array to populate
td = pd.DataFrame(index = range(len(data_clean)), columns = column_names)

for i in range(len(data_clean)):
    new_timestamp = data_clean["timestamp"][i][0:10]
    new_meter = data_clean["meter_reading"][i]
    newdata.loc[i] = [new_timestamp, new_meter]

#print(newdata)

#create test df to populate
testdata = pd.DataFrame(index = range(366), columns = column_names)
testdata = td.groupby("timestamp")["meter_reading"].median()
print(testdata)

testdata = testdata.to_frame()
ts = testdata["timestamp"].unique()
testdata.insert(0, "timestamp", ts, True)
print(testdata)

#find data from same day and average
avgdata = pd.DataFrame(index = range(366), columns = column_names)
avgdata = newdata.groupby("timestamp")["meter_reading"].mean()
print(avgdata)

avgdata = avgdata.to_frame()
ts = newdata["timestamp"].unique()
avgdata.insert(0, "timestamp", ts, True)
print(avgdata)


#create batches of three for training
def split_sequence(sequence, n_steps):
    X, y = list(),list()
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y) 

#transform df to array/list
L = avgdata["meter_reading"].astype(float).values.tolist()
seq = L
#choosing my "batch size
n_steps = 3
#split into samples
X, y = split_sequence(seq, n_steps)

for i in range(len(X)):
    print(X[i], y[i])
    
#neural network part 

# create model
model = keras.models.Sequential()
n_features = 1
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))

#compile model
model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy'])

#reshape from [samples, timestamps] into [samples, timestamps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

#fit model
history = model.fit(X, y, validation_split = 0.33, epochs = 200)

#demonstrate prediction
for i in range(len(X)):
    x_input = X[i]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input)
    print(yhat)

#list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





    
    
    
    