# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold
from python_speech_features import mfcc
from python_speech_features import delta
from prepare_data import serialize_data
from prepare_data import load_data_from_npy

npy_path = './Audio Data/npy data'
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


model = Sequential()
model.add(Dropout(0.2, input_shape=(39*41,)))
model.add(Dense(39*41,init='normal',activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # Fit the model

history = model.fit(x_train, y_train, epochs=10, batch_size=512,validation_data=(x_val,y_val))

with open('./log_history.txt') as log:
    log.write(str(history.history))

model.save('./Audio Data/model/model-test1.h5')

#plot train and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Traing and validtion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('./Traing and validtion loss.png')

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Traing acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Traing and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('Traing and validation accuracy')

results = model.evaluate(x_test, y_test)
print(results)

