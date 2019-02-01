#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

class DatasetSplit:

    emotions = ["calmness", "fear", "sadness", "disgust", "anger", "happiness"]
    participant_number = 5


    dataset_path = r'..\datasets'
    folder_general = '..\Emotion_Dataframes'

    folder_locations = []
    dataset_list = []

    def folder_creator(self):
        for index_main, participant in enumerate(range(participant_number)):

            folder_locations.append(folder_general + "\\" + ("Participant" + str(participant + 1)))

            for index,emotion in enumerate(emotions):
                name = folder_general + "\\" + ("Participant" + str(participant + 1)) + "\\" + emotion
                if not os.path.isdir(name):
                    os.makedirs(name)
                    print("Directory %s is created" %name)
                else:
                    print("Same name of Directory %s is already exists" %name)

            dataset = pd.read_csv(dataset_path + "\\" +  "record" + str(index_main) + ".csv") 
            dataset_list.append(dataset)
    
    def data_splitter(self):
        for index_main, dataset in enumerate(dataset_list):
            data_list= np.array_split(dataset, 6)
            for index, emotion in enumerate(data_list):
                file_name = folder_locations[index_main] + "\\" + emotions[index] +  "\\" + emotions[index] + "_dataframe.csv"
                emotion.to_csv(file_name, sep = 't')
    

splitter = DatasetSplit()
splitter.folder_creator()
splitter.data_splitter()
