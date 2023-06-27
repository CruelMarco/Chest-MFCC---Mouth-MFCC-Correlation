# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:43:15 2023

@author: Shaique
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping
np.set_printoptions(suppress=True)

#os.chdir("C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1/")

np.set_printoptions(suppress=True)

dir ='C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/LL/fold1'

files = os.listdir(dir)

train__mouth_path = os.path.join(dir, files[1])

train__chest_path = os.path.join(dir, files[0])

train_mouth = pd.read_csv(train__mouth_path)

train_chest = pd.read_csv(train__chest_path)

val__mouth_path = os.path.join(dir, files[3])

val__chest_path =os.path.join(dir, files[2])

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

b0 = []

b1 = []

b2 = []

b3 = []

b4 = []

for i in range(1):

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
   
    early_stopping = EarlyStopping(monitor='val_loss', patience = 40)
    history = model.fit(X_train, y_train, validation_data= (X_val, y_val), epochs=100, callbacks=[early_stopping])
    
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



    select_breath = val_chest.loc[val_chest['breath'] == 'b0']
    print (type(select_breath))
    index_b0 = select_breath.index.tolist()
    start = index_b0[0]
    end =index_b0[len(index_b0)-1]
    b0_idx = [i for i in range(start, end+1)]
    b0_mse = mse_result[b0_idx]
    b0_mse_mean = np.mean(b0_mse)
    b0.append( b0_mse_mean)
    
    select_breath_b1 = val_chest.loc[val_chest['breath'] == 'b1']
    index_b1 = select_breath_b1.index.tolist()
    start = index_b1[0]
    end =index_b1[len(index_b1)-1]
    b1_idx = [i for i in range(start, end+1)]
    b1_mse = mse_result[b1_idx]
    b1_mse_mean = np.mean(b1_mse)
    b1.append( b1_mse_mean)
    
    select_breath_b2= val_chest.loc[val_chest['breath'] == 'b2']
    index_b2 = select_breath_b2.index.tolist()
    start = index_b2[0]
    end =index_b2[len(index_b2)-1]
    b2_idx = [i for i in range(start, end+1)]
    b2_mse = mse_result[b2_idx]
    b2_mse_mean = np.mean(b2_mse)
    b2.append( b2_mse_mean)
    
    
    if 'b3' in val_chest.values:
    
        select_breath_b3= val_chest.loc[val_chest['breath'] == 'b3']
        index_b3 = select_breath_b3.index.tolist()
        start = index_b3[0]
        end =index_b3[len(index_b3)-1]
        b3_idx = [i for i in range(start, end+1)]
        b3_mse = mse_result[b3_idx]
        b3_mse_mean = np.mean(b3_mse)
        b3.append( b3_mse_mean)
    if 'b4' in val_chest.values:
       
        select_breath_b4= val_chest.loc[val_chest['breath'] == 'b4']
        index_b4 = select_breath_b4.index.tolist()
        start = index_b4[0]
        end =index_b4[len(index_b4)-1]
        b4_idx = [i for i in range(start, end+1)]
        b4_mse = mse_result[b4_idx]
        b4_mse_mean = np.mean(b4_mse)
        b4.append( b4_mse_mean)
        #b4_mse = b4_mse.append(b4_mse_mean)
        
mean_0= np.mean(b0)  
mean_1= np.mean(b1)  
mean_2= np.mean(b2)  
mean_3= np.mean(b3)  
mean_4= np.mean(b4)  
  
#test_x_path = 'C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/test/test_mfcc/pnoistor_may2023-aditis_96917e0d-VBA_before-940b-comn_bb1.csv'

#test_x_csv = pd.read_csv(test_x_path)

#test_y_path = 'C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/test/test_mfcc/pnoistor_may2023-aditis_96917e0d-LBA_before_LU-db8d-comnt.csv'
    
#test_y_csv = pd.read_csv(test_y_path)
