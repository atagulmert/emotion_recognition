import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


df = pd.read_csv("Emotion_Dataframes/Participant01/anger/s01_anger_video01")

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
        
# --------------  filtering the ecg signal -------------
ecg = df.iloc[:,0]
df.drop(["ecg"]) # removing the ecg column to replace it with filtered ecg
ecg_filt = butter_highpass_filter(ecg,0.6,290) # the sample frequency of our dataset is 290
df['ecg'] = ecg_filt.to_list() # adding filtered ecg to the dataframe as a column
