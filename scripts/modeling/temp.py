# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU, PReLU


from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import numpy as np
import pandas as pd

import time
data = pd.read_csv("./poker-hand-training-true.csv", header = None)
data_2 = pd.read_csv("./poker-hand-testing.csv",header=None)

data_2 = data_2.iloc[0:100000,:]

labels = data.iloc[:,10]
features = data.iloc[:,0:10]

labels_2 = data_2.iloc[:,10]
features_2 = data_2.iloc[:,0:10]

encoded_labels = pd.get_dummies(labels)
encoded_labels_2 = pd.get_dummies(labels_2)


final_data = pd.concat((features,encoded_labels),axis = 1)
final_data_2 = pd.concat((features_2,encoded_labels_2),axis=1)

model = Sequential()
model.add(Dense(256, input_shape = (10,)))
model.add(LeakyReLU(alpha=0.001))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.001))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.001))
model.add(Dense(10, activation='softmax'))

t = time.process_time()

train_x = final_data.iloc[:,0:10]
train_y = final_data.iloc[:,10:]

train_x_2 = final_data_2.iloc[:,0:10]
train_y_2 = final_data_2.iloc[:,10:]

rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_x,train_y,validation_data=(train_x_2,train_y_2),epochs = 25,batch_size = 32)

elapsed_time = time.process_time() - t

print("The execution time of training: ", elapsed_time)

# Optimizer comparision - 25 epochs - LEaky ReLU
# Adam: 
#       1st) loss: 0.1667 - acc: 0.9449 - val_loss: 0.2205 - val_acc: 0.9256
#       2nd) loss: 0.3025 - acc: 0.8859 - val_loss: 0.3615 - val_acc: 0.8607
#       3rd) loss: 0.2633 - acc: 0.9070 - val_loss: 0.2995 - val_acc: 0.8961
# RMSProp: 
#       1st) loss: 0.9186 - acc: 0.5828 - val_loss: 0.9190 - val_acc: 0.5766
#       2nd) val_acc: 0.8667 The execution time of training:  19.21875
#       3rd) val_acc: 0.8000 The execution time of training:  19.859375
#       4th) val_acc: 0.7667 The execution time of training:  19.953125
#       5th) val_acc: 0.7667 The execution time of training:  21.671875
# ------------------------------------------------------------------------