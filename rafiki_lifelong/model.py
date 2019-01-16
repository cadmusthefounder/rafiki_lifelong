import pickle
import numpy as np 
from os.path import isfile

from utils import pip_install, extract

class Model:

    def __init__(self, data_info, time_info):

        # Print data and time information
        info_dict = extract(data_info, time_info)
        print('Dataset budget: {0:d} seconds'.format(info_dict['time_budget']))
        print('No. of time features: {0:d}'.format(info_dict['no_of_time_features']))
        print('No. of numerical features: {0:d}'.format(info_dict['no_of_numerical_features']))
        print('No. of categorical features: {0:d}'.format(info_dict['no_of_categorical_features']))
        print('No. of mvc features: {0:d}'.format(info_dict['no_of_mvc_features']))
        print('Overall time spent: {0:5.2f} seconds'.format(info_dict['overall_time_spent']))
        print('Dataset time spent: {0:5.2f} seconds'.format(info_dict['dataset_time_spent']))  

        # Install hyperopt and lightgbm
        pip_install('hyperopt')
        pip_install('lightgbm')
        
        # Initialise stream processor
        from stream_processor import StreamSaveRetrainPredictor
        self.mdl = StreamSaveRetrainPredictor()

    def fit(self, F, y, data_info, time_info):
        '''
        This function trains the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        '''

        info_dict = extract(data_info, time_info)
        print('Overall time spent: {0:5.2f} seconds'.format(info_dict['overall_time_spent']))
        print('Dataset time spent: {0:5.2f} seconds'.format(info_dict['dataset_time_spent']))  
       
        self.mdl.partial_fit(F, y, data_info, time_info)
       
    def predict(self, F, data_info, time_info):
        '''
        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. 
        The function predict eventually returns probabilities or continuous values.
        '''
   
        info_dict = extract(data_info, time_info)
        print('Overall time spent: {0:5.2f} seconds'.format(info_dict['overall_time_spent']))
        print('Dataset time spent: {0:5.2f} seconds'.format(info_dict['dataset_time_spent']))  
        
        return self.mdl.predict(F, data_info, time_info)
      
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
        return self