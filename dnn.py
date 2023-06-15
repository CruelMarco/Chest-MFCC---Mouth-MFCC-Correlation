# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:43:15 2023

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import os

np.set_printoptions(suppress=True)

os.chdir("C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1/")

train__mouth_path = "C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1/train_mouth.csv"

train__chest_path = "C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1/train_chest.csv"

train_mouth = pd.read_csv(train__mouth_path)

train_chest = pd.read_csv(train__chest_path)

val__mouth_path = "C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1/val_mouth.csv"

val__chest_path = "C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1/val_chest.csv"

val_mouth = pd.read_csv(val__mouth_path)

val_chest = pd.read_csv(val__chest_path)

# Separate Target Variable and Predictor Variables
TargetVariable=['F1', 'F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']
Predictors=['F1', 'F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']  

X =train_mouth[TargetVariable].values

y = train_chest[Predictors].values

X_v =val_mouth[TargetVariable].values
y_v =val_chest[Predictors].values

# ### Sandardization of data ###
# from sklearn.preprocessing import StandardScaler
# PredictorScaler=StandardScaler()
# TargetVarScaler=StandardScaler()
 
# # Storing the fit object for later reference
# PredictorScalerFit=PredictorScaler.fit(X)
# TargetVarScalerFit=TargetVarScaler.fit(y)
 
# # Generating the standardized values of X and y
# X=PredictorScalerFit.transform(X)
# y=TargetVarScalerFit.transform(y)

# ### Sandardization of data ###
# from sklearn.preprocessing import StandardScaler
# PredictorScaler=StandardScaler()
# TargetVarScaler=StandardScaler()
 
# # Storing the fit object for later reference
# PredictorScalerFit=PredictorScaler.fit(X_v)
# TargetVarScalerFit=TargetVarScaler.fit(y_v)
 
# # Generating the standardized values of X and y
# X=PredictorScalerFit.transform(X_v)
# y=TargetVarScalerFit.transform(y_v)


X_train = X
y_train = y

X_val = X_v
y_val= y_v

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


model = Sequential()
#1 st layer 
model.add(Dense(12, input_shape=(12,), activation = 'sigmoid'))
    #2nd layer
model.add(Dense(100, activation='relu'))
#3rd layer
model.add(Dense(50, activation='relu'))

#model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))

#model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
    
#model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))

#model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))

model.add(Dense(12, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

model.compile(loss= "mean_squared_error" , optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=["mean_squared_error"])

history = model.fit(X_train, y_train, validation_data= (X_val, y_val), epochs=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

val = model.evaluate(X_v, y_v)

pred = model.predict(X_v)


def row_wise_mse(array1, array2):
    # Ensure both arrays have the same shape
    assert array1.shape == array2.shape, "Arrays must have the same shape"
    
    # Calculate the squared difference between the arrays
    squared_diff = (array1 - array2) ** 2
    
    # Calculate the row-wise mean squared error
    mse = np.mean(squared_diff, axis=1)
    
    # Standardize the MSE results between 0 and 1
    normalized_mse = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))
    
    return normalized_mse

mse_result = row_wise_mse(pred, y_val)

# plt.plot(history.history['mean_squared_error'])
# plt.plot(history.history['val_mean_squared_error'])
# plt.title('Mean Squared Error')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()

# ###########################################


# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasRegressor

# # Listing all the parameters to try
# Parameter_Trials={'batch_size':[20],
#                       'epochs':[200,250],
#                     'Optimizer_trial':['adam', 'rmsprop']
#                  }

# # Creating the regression ANN model
# RegModel=KerasRegressor(model, verbose=1)

# ###########################################
# from sklearn.metrics import make_scorer

# # Defining a custom function to calculate accuracy
# def Accuracy_Score(orig,pred):
#     MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
#     print('#'*70,'Accuracy:', 100-MAPE)
#     return(100-MAPE)

# custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# #########################################
# # Creating the Grid search space
# # See different scoring methods by using sklearn.metrics.SCORERS.keys()
# grid_search=GridSearchCV(estimator=RegModel, 
#                          param_grid=Parameter_Trials, 
#                          scoring=custom_Scoring, 
#                          cv=5)

# #########################################
# # Measuring how much time it took to find the best params
# import time
# StartTime=time.time()

# # Running Grid Search for different paramenters
# grid_search.fit(X,y, verbose=1)

# EndTime=time.time()
# print("########## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes')

# print('### Printing Best parameters ###')
# grid_search.best_params_
# '''
# loss = model.evaluate(X_val, y_val, verbose=0)

# print("Validation Loss:", loss)'''
# #manualgridsearch to find hyperparameters
# #def FunctionFindBestParams(X_train, y_train, X_val, y_val):
    
#     # Defining the list of hyper parameters to try
  

