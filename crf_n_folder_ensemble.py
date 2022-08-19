import sklearn_crfsuite
import pickle
import os
import pandas as pd
import glob
import numpy as np
import random
from sklearn_crfsuite.compat import BaseEstimator

class crf_n_folder_ensemble(BaseEstimator):

    def __init__(self, data_folder_path, model_path, n_fold=1):
        
        self.data_folder_path = data_folder_path # folder path to read data from
        self.model_path = model_path # folder path to save ensembled models
        self.n_fold = n_fold # number of fold that will be ensembled
        self.crf = sklearn_crfsuite.CRF( #crf model
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True)
        self.models = list() # model list that keeps the trained models
        
    def prepare_data(self):
        
        df = self.create_dataframe(self.data_folder_path)# create test dataframe
        classes = self.unique_classes(df) # return unique tag classes

        return df, classes
    

    def create_dataframe(self, data_folder):
        
        try:
            train_file_path = os.path.join(data_folder, "*.csv")
            train_files = glob.glob(train_file_path)
        except:
            raise Exception("Problem has occured while listing train files in data folder.")

        try:
            df = pd.concat(map(pd.read_csv, train_files), ignore_index=True) # read each csv and concatenate them into one
        except:
            raise Exception("An error occured while reading csv files.")

        return df
    
    def unique_classes(self, dataframe):
        
        try:
            tags = dataframe.tag.values # get tag values
        except:
            raise Exception("Tag's should be in data frame's tag column, with lower case!")
        
        classes = np.unique(tags).tolist() # get unique classes
        return classes
    
    def partition(self, X_train, y_train):
        
        if len(X_train) != len(y_train):
            raise Exception("Input X_train and y_train sizes must have same length. Please check your inputs.")

        try:
            zipped = zip(X_train, y_train)
            zipped_list = list(zipped)
        except:
            raise Exception("An error occured while zipping X_train and y_train.")
        
        try:
            random.shuffle(zipped_list)
        except:
            raise Exception("An error occured while shuffling zipped X_train and y_train.")
        
        if self.n_fold < 1 or self.n_fold > len(X_train):
            raise Exception("Constructor input n_fold must be greater than or equal to 1 and lower than X_train length.")
        
        train_list = []
        for i in range(self.n_fold):
            random_zipped = zipped_list[i::self.n_fold]
            
            try:
                X, y = zip(*random_zipped)
            except:
                raise Exception("An error occured while unzipping X_train and y_train set.")

            train_list.append((X,y))
        return train_list


    def fit(self, X_train, y_train): 

        train_list = self.partition(X_train, y_train)

        models = []
        for train_set in train_list: # for each train data in X_list
            X_train, y_train = train_set
            try:
                self.crf.fit(X_train, y_train) # train a seperate model
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

a = crf_n_folder_ensemble("data", "model", 1)

a.fit([1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14])
