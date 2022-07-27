# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:55:45 2022

@author: fizah
"""
#%%

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras import Sequential, Input, Model
import matplotlib.pyplot as plt
#%%

class ModelDevelopment:
    def  dl_model(self,output_shape,input_shape,nb_node=60,dropout_rate=0.3,activation='linear'):
                     
        model = Sequential()
        model.add(Input(shape=input_shape)) 
        model.add(LSTM(nb_node,return_sequences=(True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(nb_node)) 
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_shape,activation))
        model.summary()
        
        return model
    
class ModelEvaluation:
    def actual_predict_plot(self, actual, predicted):
        
        plt.figure()
        plt.plot(actual, color='red')
        plt.plot(predicted, color='blue')
        plt.xlabel('Days')
        plt.ylabel('New Cases')
        plt.legend(['Actual Cases','Predicted Cases'])
        plt.show()
        
        