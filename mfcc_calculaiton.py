# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:39:24 2023

@author: Spirelab
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import os
import librosa
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout, Input
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, RMSprop
from keras import optimizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Bidirectional
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten,Dropout,MaxPooling1D,Activation
from sklearn.model_selection import train_test_split
import itertools


dir = 'C:/Users/Spirelab/Desktop/mfcc_correlation/audio_dataset/audio-data_feb2023'

os.chdir(dir)

files = os.listdir(dir)

def loca_df_creator(loc_code, annot_file):
    
    loc_st_en_indices = annot_file.index[annot_file['phon'] == str(loc_code)].tolist()
    
    loc_indices = list(range(loc_st_en_indices[0]+1, loc_st_en_indices[1]))
    
    loc_annot_df = annot_file.iloc[loc_indices]
    
    loc_annot_df[["start" , "end"]] = loc_annot_df[["start" , "end"]].apply(lambda x:x * sr)
    
    st_sam = np.ceil(list(loc_annot_df["start"]))
    
    en_sam = np.ceil(list(loc_annot_df["end"]))
    
    phon_sam = list(loc_annot_df["phon"])
    
    limit = len(loc_annot_df)  # the maximum value in the list
    
    repeating_seq = itertools.chain.from_iterable(itertools.repeat(x, 2) for x in range(limit//2 + 1))
    
    wheeze_idx = list(itertools.islice(repeating_seq, limit))
    
    loc_annot_df['idx'] = wheeze_idx
    
    return(loc_annot_df, st_sam, en_sam, phon_sam )

for i in files:
    
    sub_path = os.path.join(dir, i)
    
    print(sub_path)
    
    wav_path = os.path.join(sub_path, [f for f in os.listdir(sub_path) if f.endswith('.wav') or f.endswith('.WAV')][4])
    
    annot_path = os.path.join(sub_path, [f for f in os.listdir(sub_path) if f.endswith(".txt")][0])
    
    wav_file,sr = librosa.load(wav_path, sr = None, mono = False)
    
    mouth_wav = wav_file[0]
    
    chest_wav = wav_file[1]
    
    annot_file = pd.read_csv(annot_path, sep = '\t', names= ['start' , 'end' , 'phon'], header = None)
    
    LU_annot_df, LU_st_sam, LU_en_sam, LU_phon_sam = loca_df_creator('oo' , annot_file)
    
    LU_wheeze_idx = list(LU_annot_df['idx'].unique())
    
    for j in LU_wheeze_idx:
        
        wheeze_samps_df = LU_annot_df.loc[LU_annot_df['idx'] == j]
        
        st_sam = round(list(wheeze_samps_df['start'])[0])
        
        end_sam = round(list(wheeze_samps_df['end'])[1])
        
        wheeze_chunk_mouth = mouth_wav[st_sam : end_sam]
        
        wheeze_chunk_chest = chest_wav[st_sam : end_sam]  
        
        mfcc_mouth = librosa.feature.mfcc(y = wheeze_chunk_mouth, sr = sr, n_mfcc = 13, win_length = 882, hop_length = 441)
        
        mfcc_chest = librosa.feature.mfcc(y = wheeze_chunk_chest, sr = sr, n_mfcc = 13, win_length = 882, hop_length = 441)

    RU_annot_df, RU_st_sam, RU_en_sam, RU_phon_sam = loca_df_creator('aa' , annot_file)
    
    st_list  = []
    
    end_list = []
    
    for i in RU_annot_df['idx'].unique():
        
        RU_annot_df_idx = RU_annot_df['idx'] 
    
    LL_annot_df, LL_st_sam, LL_en_sam, LL_phon_sam = loca_df_creator('ee' , annot_file)
    
    RL_annot_df, RL_st_sam, RL_en_sam, RL_phon_sam = loca_df_creator('uu' , annot_file)
    
    LU_annot_df, LU_st_sam, LU_en_sam, LU_phon_sam = loca_df_creator('oo' , annot_file)
    
    
    
    
    
    
    
    
    
    
    
    
    #breath_chunk = 
    

    
    
    
    
    
    
    
    
    
    