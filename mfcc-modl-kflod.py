# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from prepare_data import load_data_from_npy

def create_model():
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(39*41,)))
    model.add(Dense(39*41,kernal_initial='normal',activation='relu',kernal_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu',kernal_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu',kernal_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(5,activation='softmax'))

    return model

npy_path = './npy data'
train_wav_npy_filename = 'train_wav.npy'
train_tg_npy_filename = 'train_label.npy'
test_wav_npy_filename = 'test_wav.npy'
test_tg_npy_filename = 'test_label.npy'
val_wav_npy_filename = 'val_wav.npy'
val_tg_npy_filename = 'val_label.npy'

x_train = load_data_from_npy(os.path.join(npy_path,train_wav_npy_filename))
y_train = load_data_from_npy(os.path.join(npy_path,train_tg_npy_filename))
x_test = load_data_from_npy(os.path.join(npy_path,test_wav_npy_filename))
y_test = load_data_from_npy(os.path.join(npy_path,test_tg_npy_filename))
x_val = load_data_from_npy(os.path.join(npy_path,val_wav_npy_filename))
y_val = load_data_from_npy(os.path.join(npy_path,val_tg_npy_filename))

x_train = np.concatenate((x_train,x_val),axis=0)
y_train = np.concatenate((y_train,y_val),axis=0)

#define 10-fold cross validation test harness
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model,nb_epoch=20,batch_size=512)
kflod = StratifiedKFold(y=x_train,n_folds=10,shuffle=True,random_state=seed)
results = cross_val_score(model,x_train,y_train,cv=kflod)
print(results.mean())