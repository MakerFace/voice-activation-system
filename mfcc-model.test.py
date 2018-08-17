# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from python_speech_features import mfcc
from python_speech_features import delta
from prepare_data import preparing_data

train_wav_path = '../Audio Data/all_output_wav'
train_tg_path = '../Audio Data/text grid all test'
test_wav_path = '../Audio Data/spj_output_wav'
test_tg_path = '../Audio Data/text grid spj test'

x_train,y_train = preparing_data(train_wav_path,train_tg_path)
x_test,y_test = preparing_data(test_wav_path,test_tg_path)

#shuffle narray
training_data = np.hstack([x_train,y_train])
np.random.shuffle(training_data)
x_train = training_data[:,:-5]
y_train = training_data[:,-5:]


x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]


model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(39*41,)))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # Fit the model

history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512,validation_data=(x_val,y_val))

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

plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Traing acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Traing and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate(x_test, y_test)
print(results)

