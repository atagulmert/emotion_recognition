#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


#  ****************Configuration******************
#  1) Download all of the datasets from the original source
#  2) Place them in a folder named "datasets"
#  3) Place the folder "datasets" with the same location of this script
#  4) Once you run the script, "Emotion_Dataframes" folder must be created into the same location

class DatasetSplit:

    emotions = ["calmness", "fear", "sadness", "disgust", "anger", "happiness"]
    dataset_number = 0 # how many dataset files do we have
    video_number = 2   
    participant_id_index = 0
    script_path = Path().absolute() #location of our script


    dataset_path =  os.path.join(str(script_path), 'datasets') #path of the folder which datasets stay
    folder_general = os.path.join(str(script_path), 'Emotion_Dataframes')
    
     

    file_names = []
    folder_locations = []
    dataset_list = []
    splitted_datasets = {}

    
    def load_dataset(self):

        for filename in os.listdir(self.dataset_path):
            self.file_names.append(filename)
            try:
                dataset = pd.read_csv(os.path.join(self.dataset_path, filename))
                self.dataset_list.append(dataset)
            except:
                print("ERROR! %s is not exist" %(os.path.join(self.dataset_path , filename)))

        self.dataset_number = len(self.file_names)
        
        

    def create_folder(self):
        for index_main, dataset in enumerate(range(self.dataset_number)): #looping into all datasets
            file_name = self.file_names[index_main] #file name of the current dataset
            #participant_id_index = int(re.search('\d', file_name).group()) #extracting the id number of the participant from the dataset file name
            participant_id_index = self.find_first_digit(file_name)
            
            participant_id = self.find_file_id(file_name, participant_id_index) #slicing the dataset file name to extract id number of the participant4
            self.folder_locations.append(os.path.join(self.folder_general , "Participant" + participant_id))
        
            for index,emotion in enumerate(self.emotions):
                if "video02" in self.file_names[index_main]:
                    pass
                else:
                    name = os.path.join(self.folder_locations[index_main] , emotion) #the name of the folder path which we want to create
                    if not os.path.isdir(name):
                        os.makedirs(name)
                        #print("Directory %s is created" %name)

                    else:
                        #print("Same name of Directory %s is already exists" %name)  
                        pass
                        
                    
                    
    def data_splitter(self):
        video_id = 0 #initialization of video id
        file_name = ""
        for index_main, (dataset,name) in enumerate(zip(self.dataset_list, self.file_names)):
            data_split = np.array_split(dataset, 6)
            self.splitted_datasets[name] = data_split
            file_id = self.find_file_id(name, self.find_first_digit(name))
            for index, emotion in enumerate(data_split):
                extension = "s" + str(file_id).zfill(2) + "_" + self.emotions[index] + "_video0"
                file_name = os.path.join(self.folder_locations[index_main] , self.emotions[index] , extension )
                if( "video02" in name):
                    video_id = 2
                else:
                    video_id = 1
                file_name = file_name + str(video_id) + ".csv"
                emotion.to_csv(file_name,encoding='utf-8', index = False, float_format = "%.4f")
            
    
    def find_first_digit(self, name: str): #finds the first integer digit's index in a string
         for i, c in enumerate(name):
                if c.isdigit():
                    return i
                    break
                    
    def find_file_id(self, name, index):
        return name[(index):(index+2)]
                

splitter = DatasetSplit()
splitter.load_dataset()
splitter.create_folder()
splitter.data_splitter()
