# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 18:27:21 2023

@author: Spirelab
"""

import pandas as pd
import os
import numpy as np

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


dir ='C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/13 folds LL/RL'

os.chdir(dir)

folds = os.listdir(dir)

fold_wise_mse = []

fold_count = []

for j in folds:
    
    print(j)
    
    fold_dir = os.path.join(dir, j)
    
    files = os.listdir(fold_dir)

    chest_location = fold_dir.split("/")[7]
    
    mouth_csv_dir = os.path.join(fold_dir, files[3])
    
    chest_csv_dir = os.path.join(fold_dir, files[2])
    
    mouth_csv = pd.read_csv(mouth_csv_dir).drop(columns = ['Unnamed: 0.1' , 'Unnamed: 0', ])
    
    chest_csv = pd.read_csv(chest_csv_dir).drop(columns = ['Unnamed: 0.1' , 'Unnamed: 0'])
    
    breath_col = mouth_csv["breath"]
    
    breath_num = list(breath_col.unique())
    
    breath_wise_mse = []
    
    for i in breath_num:
        
        mouth_breath_mfcc = mouth_csv.loc[mouth_csv['breath'] == i].drop(columns = ['breath', 'position' , 'Sub_name']).values
        
        chest_breath_mfcc = chest_csv.loc[chest_csv['breath'] == i].drop(columns = ['breath', 'position' , 'Sub_name']).values
        
        breath_mse = np.mean(row_wise_mse(mouth_breath_mfcc, chest_breath_mfcc))
        
        breath_wise_mse.append(breath_mse)
    
    breath_mse = np.mean(breath_wise_mse)
    
    fold_count.append(j)
    
    fold_wise_mse.append(breath_mse)
    
mse_data = {'fold_number' : fold_count , 'mse' : fold_wise_mse}

mse_df = pd.DataFrame(mse_data, columns = ['fold_number' , 'mse'])    

    
    






