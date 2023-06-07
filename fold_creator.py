# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:56:09 2023

@author: Spirelab
"""

import os
import pandas as pd
import numpy


dir = 'C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/data_set_ll/mouth_ll'

all_sub_mfcc_dir = 'C:/Users/Spirelab/Desktop/mfcc_correlation/aditi_files/all_sub_mfcc'

fold_struc_dir = 'C:/Users/Spirelab/Desktop/mfcc_correlation/mfcc_data_set/fold_structure'

os.chdir(dir)

files = os.listdir(dir)

all_sub_mfcc = []

for i in files:
    
    #print(i)
    
    file_name = i[0:-4]
    
    sub_name_temp = file_name.split('-')[1]
    
    loc = file_name.split('_')[4]
    
    sub_name = sub_name_temp + '_' + loc
    
    mfcc_file_dir = os.path.join(dir, i)
    
    sub_mfcc = pd.read_csv(mfcc_file_dir)
    
    sub_mfcc = sub_mfcc.drop(['Unnamed: 0'], axis = 1)
    
    sub_mfcc["Sub_name"] = sub_name_temp
    
    name_col = sub_mfcc.pop("Sub_name")
    
    sub_mfcc.insert(0,"Sub_name", name_col)
    
    all_sub_mfcc.append(sub_mfcc)
    
all_sub_mfcc = pd.concat((all_sub_mfcc))

mfcc_name = 'all_sub_mfcc.csv'

dest_mfcc_dir = os.path.join(all_sub_mfcc_dir, mfcc_name )

all_sub_mfcc.to_csv(dest_mfcc_dir)

all_sub_mfcc_csv = pd.read_csv(dest_mfcc_dir)

all_sub_mfcc_csv = all_sub_mfcc_csv.drop(['Unnamed: 0'], axis = 1)

sub_name_list = all_sub_mfcc_csv["Sub_name"].unique()

k = 1

for j in sub_name_list:
    
    #print(j)
    
    val_mfcc = all_sub_mfcc_csv.loc[all_sub_mfcc_csv["Sub_name"] == j]
    
    os.mkdir
    
    fold_name = 'Fold' + '_' + str(k)
    
    fold_dir = os.path.join(fold_struc_dir, fold_name)
    
    if not os.path.exists(fold_dir):
        
        os.mkdir(fold_dir)
        
    else:
        
        print("Folder already exists")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    