# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:27:16 2022

@author: fizah
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from tensorflow.keras.losses import MeanAbsolutePercentageError
from modules_covid import ModelDevelopment,ModelEvaluation
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import pickle
import os

pd.set_option('display.max_columns', None)
#%% Constants

CSV_PATH_TRAIN = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')

MMS_PATH = os.path.join(os.getcwd(),'saved_files','mms_train.pkl')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
#%% Data Loading

df_train = pd.read_csv(CSV_PATH_TRAIN, na_values = [' ','?'])
df_test = pd.read_csv(CSV_PATH_TEST, na_values = '?')
#%% Data Inspection

df_train.head() 
df_train.tail()
df_train.columns
df_train.info()
df_train.isna().sum()
# NaNs in df_train['cases_new'] is represented with '?' and blank cells
# df_train has NaNs in several columns starting from 25/1/2020 - 31/12/2020
# df_train has 12 NaN in cases_new, which needs to be interpolated
df_train[df_train['cases_new'].isna()] #
# One in NaN in year 2020, 5 NaNs in 2021 and most of the NaNs are missing one day,
# but 14/4/2021 & 15/4/2021 are missing for two consecutive days.
# Interpolation:
# Linear:a straight line is drawn between two points on a graph to determine the other unknown values. 
# This simple method could results in inaccurate estimates, especially for 2 consecutive missing values.
# Therefore it is better to use polynomial interpolation method for this data.

df_test.head() 
df_test.tail()
df_test.columns
df_test.info()
df_test.isna().sum()
# df_test has one NaN in cases_new, which needs to be interpolated, at index 60


#%% Data Cleaning

# Interpolating NaN in df_train (polynomial interpolation)
df_train['cases_new'] = df_train['cases_new'].interpolate(method='polynomial', order=2)
df_train['cases_new'].isna().sum()
# number of new cases cannot be in float (Change to integer)
df_train['cases_new'] = np.floor(df_train['cases_new']).astype('int64')

# view 50 data 
df_disp=df_train[251:301]
plt.figure()
plt.plot(df_disp['cases_new'])
plt.title('New Cases from day 250 - 300 with polynomial interpolation')
plt.show()

# Interpolating NaN in df_test
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial', order=2)
df_test['cases_new'].isna().sum()
# number of new cases cannot be in float (Change to integer)
df_test['cases_new'] = np.floor(df_test['cases_new']).astype('int64')

test_disp=df_test[49:69]
plt.figure()
plt.plot(test_disp['cases_new'])
plt.title('New Cases from day 50 to 70 with polynomial interpolation')
plt.show()

#%% Feature Selection

X = df_train['cases_new']
# Scaling values to between 0-1:
mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))


with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)
    
#%% Making a list of sample size for X and y train

win_size = 30
X_train = []
y_train = []

for i in range(win_size, len(X)): # from 30 to 680
    X_train.append(X[i-win_size:i]) # sample 1: 0-30, sample 2:1-31, and so on...
    y_train.append(X[i]) # sample 1: day 31, sample 2: day 32, and so on...

X_train = np.array(X_train) # in rank 3
y_train = np.array(y_train)

#%% For df_test

# combining the cases_new column from both dataframes
df_concat = pd.concat((df_train['cases_new'],df_test['cases_new']))

# Finding the length of days  to be used with together with df_test
length_days = win_size + len(df_test)
input_days = df_concat[-length_days:]

Xtest = mms.transform(np.expand_dims(input_days,axis=-1))

# Making a list of sample size for X and y test
X_test=[]
y_test = [] 

for i in range(win_size, len(Xtest)): 
    X_test.append(Xtest[i-win_size:i]) 
    y_test.append(Xtest[i])

X_test = np.array(X_test) # in rank 3
y_test = np.array(y_test)

#%% Model Development

md = ModelDevelopment()
model = md.dl_model(input_shape = (30,1),output_shape=1,nb_node=60,
                    dropout_rate=0.3,activation='linear')

plot_model(model,show_shapes=True,show_layer_names=True)
#%% 
model.compile(optimizer='adam',loss=['mse'],
                   metrics=['mean_absolute_error','mse'])

#callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
early_callback = EarlyStopping(monitor='mse',patience=5,verbose=1)

#%% Model training
hist = model.fit(X_train,y_train, epochs=80,
                 callbacks=[tensorboard_callback, early_callback])

#%% Model evaluation
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['mse'])
plt.title('Training MSE')
plt.show()

#%% Predict new cases
predicted_new_cases = model.predict(X_test)

#%% Plotting to see the actual cases vs predicted cases

me = ModelEvaluation()

# Plotting scaled data for actual vs predicted cases
me.actual_predict_plot(y_test, predicted_new_cases)

#visualizing unscaled data for actual vs predicted cases
actual_new_cases = mms.inverse_transform(y_test) #inverse
predicted_cases = mms.inverse_transform(predicted_new_cases)

me.actual_predict_plot(actual_new_cases, predicted_cases)

#%% error values

print('MAE: ', mean_absolute_error(y_test,predicted_new_cases)) 
print('MSE: ',mean_squared_error(y_test,predicted_new_cases))
print('MAPE: ',mean_absolute_percentage_error(actual_new_cases,predicted_cases))

# Testing with MAPE formula given
mape_formula = (mean_absolute_error(y_test,predicted_new_cases)/(sum(abs(y_test)))* 100)
print('MAPE Formula: ',mape_formula)
