import sklearn_crfsuite
import pickle
import os
import splitfolders
import random
import shutil
import pandas as pd
import glob
import numpy as np
from sklearn_crfsuite.compat import BaseEstimator

class crf_n_folder_ensemble(BaseEstimator):

    def __init__(self, data_folder_path=None, model_path=None, destination_folder_path="destination", output_folder_path="output", train_data_ratio=0.8, validation_data_ratio = 0.0, test_data_ratio=0.2):
        
        self.data_folder_path = data_folder_path # folder path to read data from
        self.model_path = model_path # folder path to save ensembled models
        self.destination_folder_path = destination_folder_path # path for seperated folder 
        self.output_folder_path = output_folder_path # folder that created after train test function 
        self.train_data_ratio = train_data_ratio
        self.validation_data_ratio = validation_data_ratio
        self.test_data_ratio = test_data_ratio
        self.crf = sklearn_crfsuite.CRF( #crf model
                algorithm='lbfgs',
                # c= 0.1,
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True)
        self.models = list() # model list that keeps the trained models
        
    # comment eklenecek
    def prepare_data(self, divide_data=False, num_splits=6):

        split_folders_path = self.data_folder_path # default split_folders path
        if divide_data == True: # if user wanted to seperate folders into number of split class
            self.seperate_folders(num_splits=num_splits) 
            split_folders_path = self.destination_folder_path # split folders path become output of seperate folders
        
        self.split_folders(split_folders_path) #train test split our data in split_folder_path

        train_df = self.create_train_dataframes() # create train dataframe list
        test_df = self.create_test_dataframe() # create test dataframe
        classes = self.unique_classes(test_df) # return unique tag classes

        return train_df, test_df, classes
    
    def seperate_folders(self, num_splits):

        if self.data_folder_path == None:
            raise Exception("There is no input path")

        files = os.listdir(self.data_folder_path) # list files in data folder
        num_files = len(files) # number of files in data folder
        
        if num_files == 0:
            raise Exception("Path is Empty!")

        try:
            os.mkdir(self.destination_folder_path) # create destination folder
        except:
            raise Exception("Problem has occured while creating destination folder.")
        
        if num_splits < 1 or num_splits > num_files: # check wheter num split input is valid 
            raise Exception("Number of split should be between 1 and number of files in folder!")
        
        modulus = num_files % num_splits # take modulus of number of files by number of splits so that at the end there shouldn't be any file left in the data folder
        
        if modulus < 0 or modulus > num_splits:
            raise Exception("An error occured while calculating modulus please check your inputs.")

        files_per_folder = int(num_files/num_splits) # calculate number of files per folder
        
        if files_per_folder < 1 or files_per_folder > num_files:
            raise Exception("An error occured while calculating files per folder please check your inputs.")
        
        for i in range(num_splits): 
            
            try:
                os.mkdir(self.destination_folder_path + "/" + str(i)) # create sub folder
            except:
                raise Exception("Problem has occured while creating splitted folder.")
            
            for _ in range(files_per_folder): 
                random_file = random.choice(os.listdir(self.data_folder_path)) # take random file from data folder
                try:
                    shutil.move(self.data_folder_path + "/" + random_file, self.destination_folder_path + f"/{i}/" + random_file) # move file from source to destination folder
                except:
                    raise Exception("An error occured while moving data from source to destination folder please check your input paths.")
        
        folder = 0 # folder name for remaining data in source folder after seperation
        for _ in range(modulus):
            
            random_file = random.choice(os.listdir(self.data_folder_path)) # random choice from remaining files
            
            try:
                shutil.move(self.data_folder_path + "/" + random_file, self.destination_folder_path + f"/{folder}/" + random_file)
            except:
                    raise Exception("An error occured while moving data from source to destination folder please check your input paths.")
            
            folder += 1 # increment folder name so that each folder has equal files
            

    def split_folders(self, input_folder):
    
        total_ratio = self.train_data_ratio + self.validation_data_ratio + self.test_data_ratio # total ratio should be 1

        if total_ratio != 1.0:
            raise Exception("Sum of Train, Test and Validation split ratios must be equal to 1")

        try:
            splitfolders.ratio(input_folder, output=self.output_folder_path, seed=1337, ratio=(self.train_data_ratio, self.validation_data_ratio, self.test_data_ratio)) #train test split seperated data into classes
        except:
            raise Exception("An error occured while train test splitting folders.")

    def create_train_dataframes(self, data_folder="output"):
        
        train_dataframes = []
        try:
            train_classes = os.listdir(data_folder + "/train") # list train data classes
        except:
            raise Exception("Problem has occured while creating datframe.")
        
        for train_class in train_classes: # for each classes
            class_folders = os.path.join(data_folder + "/train/" + train_class, "*.csv") # takes only csv files in the folder
            class_files = glob.glob(class_folders) # get the file paths

            try:
                train_df = pd.concat(map(pd.read_csv, class_files), ignore_index=True) # read each csv and concatenate them into one
            except:
                raise Exception("An error occured while reading csv files.")

            train_dataframes.append(train_df) # append each class's train data frame to list
        return train_dataframes
    
    def create_test_dataframe(self, data_folder="output"):
        test_dataframes = []
        test_classes = os.listdir(data_folder + "/test") # list the test classes
        
        for test_class in test_classes: # for each classes
            class_folders = os.path.join(data_folder + "/test/" + test_class, "*.csv") # takes only csv files in folder
            class_files = glob.glob(class_folders) # get the file paths

            try:
                test_df = pd.concat(map(pd.read_csv, class_files), ignore_index=True) # read each csv and concatenate them into one
            except:
                raise Exception("An error occured while reading csv files.")

            test_dataframes.append(test_df) # append each class's test data frame to list
        
        test_dataframe = pd.concat(test_dataframes, ignore_index=True) # concatenate each dataframe to get one test dataframe
        return test_dataframe
    
    def unique_classes(self, test_dataframe):
        try:
            y_test = test_dataframe.tag.values # get tag values
        except:
            raise Exception("Tag's should be in data frame's tag column, with lower case!")
        
        classes = np.unique(y_test).tolist() # get unique classes
        return classes

    def fit(self, X_list, y_list): 

        if len(X_list) != len(y_list): # input list must have same length
            raise Exception('Input lengths must be equal.')
        
        models = []
        for i in range(len(X_list)): # for each train data in X_list
            try:
                self.crf.fit(X_list[i], y_list[i]) # train a seperate model
            except:
                raise Exception("An error occured while training crf model")
            models.append(self.crf) # append trained model list
        
        self.models = models 

    def predict(self, X_test):

        if len(self.models) == 0: # check wheter there is a model in instance's model list
            raise Exception("There is no model trained, try to load trained model or train a new model before predict.")

        predictions = []
        for model in self.models: # for each model in model list
            prediction = model.predict(X_test) # take the model's prediction
            predictions.append(prediction) # append it to the predictions list

        y_pred = self.compare_model_results(predictions) # compare each model's prediction
        return y_pred

    def compare_model_results(self, predictions):

        if len(predictions) == 0:
            raise Exception("Predictions cannot be empty, it should take list contains each model's prediction.")
        
        y_pred = list()
        model_number = len(predictions) # number of models used in prediction
        sentence_number = len(predictions[0]) # number of sentence of input
        
        for sentence_index in range(sentence_number):
            sentence_prediction = list()
            word_number = len(predictions[0][sentence_index]) # number of words for each snetence
            for word_index in range(word_number):
                result_dictionary = dict()
                for model_index in range(model_number): # for each model
                    prediction = predictions[model_index][sentence_index][word_index] # get the prediction for word
                    result_dictionary[prediction] = result_dictionary.get(prediction, 0) + 1 # add the prediction to result dict if prediction exist increment it's possibility else initialize it with 1
                
                ensemble_tag = max(result_dictionary, key=result_dictionary.get) # get max probability tag for word
                sentence_prediction.append(ensemble_tag) # append it to the sentence predictions
            y_pred.append(sentence_prediction) # append sentence prediction to final prediction

        return y_pred
    
    def load_from_pickle(self, path=None):
        if path == None:
            path = self.model_path
        
        try:
            model = pickle.load(open(path, "rb")) # read selected model
        except:
            raise Exception("Cannot load pickle model, Check the model's path.")
        
        if isinstance(model, list):
            self.models = model
        else:
            self.models.append(model)
    
    def save_model(self):
        
        if len(self.models) == 0:
            raise Exception("There is no model to be saved")
        try:
            with open(f"{self.model_path}.pkl", "wb") as pck:
                pickle.dump(self.models, pck)
        except:
            raise Exception("Cannot save the model!")