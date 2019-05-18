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
from sklearn.metrics import confusion_matrix

import time

import os
import os.path
from pathlib import Path

from pandas import Series
from sklearn.preprocessing import MinMaxScaler

#np.random.seed(1337)

_script_path = Path().absolute() #location of our script
_dataset_folder_name = 'Filtered_Dataframes'
_dataset_folder_path = os.path.join(str(_script_path), _dataset_folder_name)

_file_names = []
_folder_locations = []  
_dataset_list = []

for dirpath, dirnames, filenames in os.walk(_dataset_folder_path):
    for filename in [f for f in filenames if f.endswith(".csv")]:
        location = os.path.join(dirpath, filename)
        _folder_locations.append(location)
        _file_names.append(filename)

for i,location in enumerate(_folder_locations):
    temp_df = pd.read_csv(location,engine='python')
    values = temp_df.iloc[:,0].values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values)
    normalized = scaler.transform(values)
    temp_df = temp_df.drop(['ecg'],axis = 1)
    temp_df['ecg'] = normalized
    temp_df = temp_df.truncate(after = 17999)
    temp_df['participant_no'] = i
    if "anger" in location:
        temp_df['emotion'] = 0
    elif "calmness" in location:
        temp_df['emotion'] = 1
    elif "disgust" in location:
        temp_df['emotion'] = 2
    elif "fear" in location:
        temp_df['emotion'] = 3
    elif "happiness" in location:
        temp_df['emotion'] = 4
    elif "sadness" in location:
        temp_df['emotion'] = 5
    unc_columns = ['hr','spo2','timest','temp','gsr']
    temp_df = temp_df.drop(unc_columns,axis=1)
    _dataset_list.append(temp_df)

_dataset = pd.concat(_dataset_list,axis=0)
_dataset.index = range(0,len(_dataset))
_dataset = _dataset.sample(frac=1).reset_index(drop=True)

train_x = _dataset.iloc[:,0:1]
#train_x = train_x.drop(['participant_no'],axis=1)
train_y = _dataset.iloc[:,2:]

print('Row count= ', len(_dataset))

print(train_x.shape)
print(train_y.shape)

temp_x = train_x.copy()
temp_x = temp_x.iloc[3600000:,:]
temp_x = temp_x.values.reshape(112,18000,1)
temp_y = train_x.copy()
temp_y = temp_y.iloc[3600000:,:]
temp_y = temp_y.values.reshape(112,18000)

train_x = train_x.values.reshape(312,18000,1)
train_y = train_y.values.reshape(312,18000)
trunc_train_y = train_y[:,:1]

enc = OneHotEncoder(handle_unknown='ignore')
trunc_train_y = enc.fit_transform(trunc_train_y)

print(trunc_train_y.shape)
print(trunc_train_y)

model = Sequential()
#model.add(Flatten())
model.add(Conv1D(30,700,activation='relu',input_shape=(18000,1)))
model.add(Conv1D(40,700))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(4))
model.add(Flatten())
model.add(Dense(6,activation='softmax'))

print(model.summary())


rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
sgd = keras.optimizers.SGD(lr=0.000001, clipvalue=0.5)
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(train_x,trunc_train_y,epochs = 20, batch_size = 64, validation_split=0.33, shuffle=False)
prediction = model.predict(temp_x)
