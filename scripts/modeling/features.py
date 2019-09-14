from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, Flatten, BatchNormalization
from keras.optimizers import SGD

from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn import preprocessing
import numpy as np
import pandas as pd
from scipy import stats

import time

import os
import os.path
from pathlib import Path

from pandas import Series
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('emo_features.csv',engine='python')

unwanted_col = ['Ecategory (1:positive, 2:negative)'
,'num'
,'Participant'
,'likelihood'
,'observation'
,'StdResid'
,'BIC'
,'AIC'
,'cp11'
,'cp12'
,'cp13'
,'cp21'
,'cp22'
,'cp23'
,'cp31'
,'cp32'
,'cp33'
,'tpeig11'
,'tpeig22'
,'tpeig33'
]
df = df.drop(unwanted_col,axis=1)

train_x = df.iloc[:,1:]
train_y = df.iloc[:,:1]

train_y_enc = pd.get_dummies(train_y['Emotion(1:calmness, 2:fear,3:sadness,4:disgust,5:anger,6:happiness)'])

model = Sequential()
model.add(Dense(128,input_shape=(59,)))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('sigmoid'))
print(model.summary())

rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0, nesterov=False)
#sgd = keras.optimizers.SGD(lr=0.000001, clipvalue=0.5)
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics=['acc'])
model.fit(train_x,train_y_enc,epochs = 50, batch_size = 32,validation_split=0.33)
