# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:33:19 2023

@author: Shaique
"""


import os
import shutil
 
control_dir = "C:/Users/Spirelab/Desktop/mfcc_correlation/mfcc_data_set/controls/controls"

control_mfcc_dir = "C:/Users/Spirelab/Desktop/mfcc_correlation/mfcc_data_set/control_mfcc"

os.chdir(control_mfcc_dir)

subjects = os.listdir(control_dir)

for i in subjects:
    
    subject_dir = os.path.join(control_dir, i)
        
    mfcc_folder_names = os.path.join(control_mfcc_dir, i)
        
    mfcc_dirs = os.makedirs(mfcc_folder_names)
    
    sub_dir = os.path.join(control_mfcc_dir, i)
    
    csv_files = [f for f in os.listdir(subject_dir) if f.endswith(".csv")]
    
    for j in csv_files:
        
        csv_file_dir = os.path.join(subject_dir, j)
        
        shutil.copy(csv_file_dir, sub_dir)
        
        
    
    