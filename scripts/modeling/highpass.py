import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import os.path
from pathlib import Path

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

_script_path = Path().absolute() #location of our script
_dataset_folder_name = 'Emotion_Dataframes'
_dataset_folder_path = os.path.join(str(_script_path), _dataset_folder_name)
_file_names = []
_folder_locations = []
_dataset_list = [] 
    


def list_all_files():
        for dirpath, dirnames, filenames in os.walk(_dataset_folder_path):
            for filename in [f for f in filenames if f.endswith(".csv")]:
                location = os.path.join(dirpath, filename)
                _folder_locations.append(location)
                _file_names.append(filename)
                
def load_datasets():
        for location in _folder_locations:
            _dataset_list.append(pd.read_csv(location))
    
            
def apply_filter():
#edit this method to implement your own filter
    filtered_dataset_list = []        
    fs = 290
    if _dataset_list:
        for data in _dataset_list:
            ecg = data.iloc[:,0]
            result = butter_highpass_filter(ecg,10,fs)
            temp = data.copy()
            temp.drop(["ecg"], axis = 1)
            temp['ecg'] = result.tolist()
            filtered_dataset_list.append(temp)
             
    return filtered_dataset_list


list_all_files()
load_datasets()
filtered_dataset = apply_filter()

'''
# ----------------- ploting part ------------------
i = 6
df_org = _dataset_list[i]
timest_org = df_org.iloc[:,-1]
ecg_org = df_org.iloc[:,0]

df = filtered_dataset[i]
timest = df.iloc[:,-1]
ecg = df.iloc[:,0]

from pylab import rcParams
rcParams['figure.figsize'] = 25.5, 10.5


plt.figure(1)
plt.subplot(211)
plt.plot(timest,ecg)

plt.subplot(212)
plt.plot(timest_org,ecg_org)
plt.show()
'''
