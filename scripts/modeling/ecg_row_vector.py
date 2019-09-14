from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import keras
from keras.layers import Input, Dense, LeakyReLU
from keras.optimizers import SGD
from keras.layers import TimeDistributed,Embedding

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import numpy as np
import pandas as pd

import time

import os
import os.path
from pathlib import Path

_script_path = Path().absolute() #location of our script
_dataset_folder_name = 'Filtered_Dataframes'
_dataset_folder_path = os.path.join(str(_script_path), _dataset_folder_name)

_file_names = []
_folder_locations = []
_dataset_list = []
_ecg_columns = []


for dirpath, dirnames, filenames in os.walk(_dataset_folder_path):
    for filename in [f for f in filenames if f.endswith(".csv")]:
        location = os.path.join(dirpath, filename)
        _folder_locations.append(location)
        _file_names.append(filename)
        

for location in _folder_locations:
    _dataset_list.append(pd.read_csv(location))
    
    
#min_max_scaler = preprocessing.MinMaxScaler()
for dataset in _dataset_list:
    dataset = dataset.truncate(after = 2999)
    ecg = dataset.iloc[:,0]
 #   x = ecg.values.astype(float)
 #   np_scaled = min_max_scaler.fit_transform(x.reshape(-1,1))
 #   ecg_normalized = pd.DataFrame(np_scaled)
    ecg = ecg.to_frame().T
  
    _ecg_columns.append(ecg)
    
'''
anger = 0
calmness = 1
disgust = 2
fear = 3
happiness = 4
sadness  = 5
'''


label_list = []
label = -1
for filename in _file_names:
    if "anger" in filename:
        label = 0
    elif "calmness" in filename:
        label = 1
    elif "disgust" in filename:
        label = 2
    elif "fear" in filename:
        label = 3
    elif "happiness" in filename:
        label = 4
    elif "sadness" in filename:
        label = 5
    
    label_list.append(label)
    
np_emotion = np.array(label_list)
emotion = pd.DataFrame(np_emotion)
emotion.columns = ["emotion"]

encoded_emotion = pd.get_dummies(emotion.emotion)

    
temp_list = []  
for i in range(len(_ecg_columns)):        
    temp_list.append(pd.DataFrame(_ecg_columns[i].values,index=[0]))

ecg = pd.concat(temp_list)
ecg.index = range(len(_ecg_columns))


final_dataset = pd.concat([ecg,encoded_emotion], axis = 1, sort=False)
final_dataset = final_dataset.sample(frac=1)


    

train_x = final_dataset.iloc[0:200,0:3000]
train_y = final_dataset.iloc[0:200,3000:]

test_x = final_dataset.iloc[200:,0:3000]
test_y = final_dataset.iloc[200:,3000:]

train_x = train_x.values.reshape(200,3000,1)
#train_y = train_y.values.reshape(1,312,6)
model = Sequential()
model.add(LSTM(32,input_shape=(3000,1),return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=False))
model.add(Dense(6,activation='sigmoid'))


rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(train_x,train_y,epochs = 20, validation_split = 0.33,verbose=1)
score = model.evaluate(train_x,test_x,batch_size=32)
print('Score' + score)