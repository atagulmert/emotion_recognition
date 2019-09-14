from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import numpy as np
import pandas as pd

import time

import os
import os.path
from pathlib import Path

#np.random.seed(1337)

_script_path = Path().absolute() #location of our script
_dataset_folder_name = 'Filtered_Dataframes'
_dataset_folder_path = os.path.join(str(_script_path), _dataset_folder_name)

_file_names = []
_folder_locations = []
_dataset_list = []
_ecg_columns = []
_gsr_columns = []


for dirpath, dirnames, filenames in os.walk(_dataset_folder_path):
    for filename in [f for f in filenames if f.endswith(".csv")]:
        location = os.path.join(dirpath, filename)
        _folder_locations.append(location)
        _file_names.append(filename)
        

for location in _folder_locations:
    _dataset_list.append(pd.read_csv(location))
    
    
min_max_scaler = preprocessing.MinMaxScaler()
for dataset in _dataset_list:
    dataset = dataset.truncate(after = 17999)
    ecg = dataset.iloc[:,0]
    gsr = dataset.iloc[:,1]
    x = ecg.values.astype(float)
    y = gsr.values.astype(float)
    np_scaled = min_max_scaler.fit_transform(x.reshape(-1,1))
    gsr_np_scaled = min_max_scaler.fit_transform(y.reshape(-1,1))
    gsr_normalized = pd.DataFrame(gsr_np_scaled)
    ecg_normalized = pd.DataFrame(np_scaled)
    #ecg = ecg.to_frame().T
    ecg = ecg_normalized.T
    gsr = gsr_normalized.T
  
    _ecg_columns.append(ecg)
    _gsr_columns.append(gsr)
    
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
        label = 2
    elif "disgust" in filename:
        label = 1
    elif "fear" in filename:
        label = 0
    elif "happiness" in filename:
        label = 2
    elif "sadness" in filename:
        label = 1
    
    label_list.append(label)
    
np_emotion = np.array(label_list)
emotion = pd.DataFrame(np_emotion)
emotion.columns = ["emotion"]

encoded_emotion = pd.get_dummies(emotion.emotion) 

    
temp_list = []  
temp_list_2 = []
for i in range(len(_ecg_columns)):        
    temp_list.append(pd.DataFrame(_ecg_columns[i].values,index=[0]))
    temp_list_2.append(pd.DataFrame(_gsr_columns[i].values,index=[0]))

ecg = pd.concat(temp_list)
gsr = pd.concat(temp_list_2)
gsr.index = range(len(_gsr_columns))
ecg.index = range(312,312+len(_ecg_columns))


ecg_encoded_emotion = encoded_emotion.copy()
ecg_encoded_emotion.index = range(312,312+len(_ecg_columns))
ecg_dataset = pd.concat([ecg,ecg_encoded_emotion], axis = 1, sort=False)
gsr_dataset = pd.concat([gsr,encoded_emotion], axis = 1, sort = False)
final_dataset = pd.concat([gsr_dataset,ecg_dataset],axis = 0)
final_dataset = final_dataset.sample(frac=1)




train_x = final_dataset.iloc[:,0:18000]
train_y = final_dataset.iloc[:,18000:]


model = Sequential()
model.add(Dense(16, input_shape = (18000,)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(3, activation='sigmoid'))


rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.9, decay=0.0, nesterov=False)
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(train_x,train_y,epochs = 300, batch_size =24, validation_split=0.33, shuffle=False)



'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33)


from sklearn.ensemble import RandomForestClassifier
import sklearn
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

from sklearn.metrics import accuracy_score

print("Train Accuracy :: ", accuracy_score(y_train, rf.predict(x_train)))
print("Test Accuracy  :: ", accuracy_score(y_test, predictions))

cm = sklearn.metrics.confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))
print(cm)
'''