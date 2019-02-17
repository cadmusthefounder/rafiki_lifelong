import pickle
import numpy as np 
import pandas as pd
from os.path import isfile

from utils import pip_install, extract, print_data_info, print_time_info, onehot_sparse
from constants import Algo
from data_processor import DataProcessor
from sampler import Sampler

pip_install('hyperopt')
pip_install('lightgbm')
pip_install('scipy')

from hyperopt import hp
from hyperparameters_tuner import HyperparametersTuner
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from stream_processor_old import *

params = {
    'algo': Algo.ORIGINAL
}

class Model:

    def __init__(self, data_info, time_info):

        # Print data information
        info_dict = extract(data_info, time_info)
        print_data_info(info_dict)

        # # Install hyperopt and lightgbm
        # pip_install('hyperopt')
        # pip_install('lightgbm')

        # Settings
        self._dataset_budget_threshold = 0.8
        self._max_train_data = 400000
        self.batch_size = 100000
        self.delta_n_estimators = 100
        self.delta_num_leaves = 20
        self.delta_learning_rate = 0.005
        self.delta_max_depth = 1
        self.delta_feature_fraction = 0.1
        self.delta_bagging_fraction = 0.1
        self.delta_bagging_freq = 1
        self.max_evaluation = 30    

        self._train_data = np.array([])
        self._train_labels = np.array([])
        self._transformed_train_data = np.array([])
        self.best_hyperparams = {}
        self._classifier = None
        self._classifier2 = None
        self._data_processor = DataProcessor(info_dict)
        self._sampler = Sampler()

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
        print_time_info(info_dict)

        if params['algo'] == Algo.OLD_CODE:
            return self.mdl.partial_fit(F, y, data_info, time_info)
        elif params['algo'] == Algo.ORIGINAL:
            return self._original_fit(F, y, info_dict)
        elif params['algo'] == Algo.FACEBOOK_LR:
            return self._facebook_lr_fit(F, y, info_dict)

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
        print_time_info(info_dict) 
        
        if params['algo'] == Algo.OLD_CODE:
            return self.mdl.predict(F, data_info, time_info)
        elif params['algo'] == Algo.ORIGINAL:
            return self._original_predict(F, info_dict)
        elif params['algo'] == Algo.FACEBOOK_LR:
            return self._facebook_lr_predict(F, info_dict)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
        return self

    def _original_fit(self, F, y, info_dict):
        data = self._convert_nan_to_num(F, info_dict)
        if self._data_processor.is_uninitialized:
            self._data_processor.preprocess(data)

        sampled_data, sampled_labels = self._sampler.majority_undersampling(data, y)

        if self._train_data.size == 0 and self._train_labels.size == 0:
            self._train_data = sampled_data
            self._train_labels = sampled_labels
        else:
            self._train_data = np.concatenate((self._train_data, sampled_data), axis=0)
            self._train_labels = np.concatenate((self._train_labels, sampled_labels), axis=0)

    def _original_predict(self, F, info_dict):
        data = self._convert_nan_to_num(F, info_dict)
        if self._has_sufficient_time(info_dict) or self._classifier is None:
            self._data_processor.preprocess(data)
            self._data_processor.prepare_frequency_map()

            current_train_data = self._train_data
            current_train_labels = self._train_labels

            print('self._train_data.shape: {}'.format(self._train_data.shape))
            print('self._train_labels.shape: {}'.foramt(self._train_labels.shape))

            if self._too_much_training_data():
                remove_percentage = 1.0 - (float(self._max_train_data) / self._train_data.size)
                current_train_data, current_train_labels = self._sampler.random_sample_in_order(self._train_data, \
                                                                                                self._train_labels.reshape(-1,1), \
                                                                                                remove_percentage)
                print('current_train_data.shape: {}'.format(current_train_data.shape))
                print('current_train_labels: {}'.format(current_train_labels.shape))
                # self._train_data, self._train_labels = current_train_data, current_train_labels

            self._transformed_train_data = self._data_processor.transform_data(current_train_data)
            self._transformed_train_labels = current_train_labels
            if not self.best_hyperparams:
                self._find_best_hyperparameters()
            
            self._classifier = LGBMClassifier(random_state=20, min_data=1, min_data_in_bin=1)
            self._classifier.set_params(**self.best_hyperparams) 
            self._classifier.fit(self._transformed_train_data, self._transformed_train_labels.ravel())
        
        if data.shape[0] <= self.batch_size: ### if it is relatively small array
            probs = self._classifier.predict_proba(self._data_processor.transform_data(data))[:,1]
            return probs
        else:
            print(156)
            print('BATCH')
            print('data.shape: {}'.format(data.shape))
            results = np.array([]) ## for chunking results to handle memory limit
            for i in range(0, data.shape[0], self.batch_size):
                Xsplit = data[i:(i+self.batch_size),:]
                results = np.append(results,self._classifier.predict_proba(self._data_processor.transform_data(Xsplit))[:,1])
                del Xsplit

            print('results.shape: {}'.format(results.shape))
            print('resutls.transposed.shape: {}'.format(results.T.shape))
            return results.T
        return []

    def _facebook_lr_fit(self, F, y, info_dict):
        data = self._convert_nan_to_num(F, info_dict)
        if self._data_processor.is_uninitialized:
            self._data_processor.preprocess(data)

        sampled_data, sampled_labels = self._sampler.majority_undersampling(data, y)

        if self._train_data.size == 0 and self._train_labels.size == 0:
            self._train_data = sampled_data
            self._train_labels = sampled_labels
        else:
            self._train_data = np.concatenate((self._train_data, sampled_data), axis=0)
            self._train_labels = np.concatenate((self._train_labels, sampled_labels), axis=0)
    
    def _facebook_lr_predict(self, F, info_dict):
        data = self._convert_nan_to_num(F, info_dict)
        if self._has_sufficient_time(info_dict) or self._classifier is None or self._classifier2 is None:
            self._data_processor.preprocess(data)
            self._data_processor.prepare_frequency_map()

            current_train_data = self._train_data
            current_train_labels = self._train_labels
            if self._too_much_training_data():
                remove_percentage = 1.0 - (float(self._max_train_data) / self._train_data.size)
                current_train_data, current_train_labels = self._sampler.random_sample_in_order(self._train_data, \
                                                                                                self._train_labels.reshape(-1,1), \
                                                                                                remove_percentage)
                # self._train_data, self._train_labels = current_train_data, current_train_labels

            self._transformed_train_data = self._data_processor.transform_data(current_train_data)
            self._transformed_train_labels = current_train_labels
            if not self.best_hyperparams:
                self._find_best_hyperparameters()
            
            self._classifier = LGBMClassifier(random_state=20, min_data=1, min_data_in_bin=1)
            self._classifier.set_params(**self.best_hyperparams) 
            self._classifier.fit(self._transformed_train_data, self._transformed_train_labels.ravel())

            probs = self._classifier.predict(self._transformed_train_data, pred_leaf=True)
            new_probs = onehot_sparse(probs)
            self._classifier2 = LogisticRegression()
            self._classifier2.fit(new_probs, self._transformed_train_labels.ravel())
            del probs
            del new_probs
        
        if data.shape[0] <= self.batch_size: ### if it is relatively small array
            probs = probs = self._classifier.predict(self._data_processor.transform_data(data), pred_leaf=True)
            new_probs = onehot_sparse(probs)
            actual_probs = self._classifier2.predict_proba(new_probs)[:,1]
            return actual_probs
        else:
            print(204)
            print('BATCH')
            print('data.shape: {}'.format(data.shape))
            results = np.array([]) ## for chunking results to handle memory limit
            for i in range(0, data.shape[0], self.batch_size):
                Xsplit = data[i:(i+self.batch_size),:]
                probs = probs = self._classifier.predict(self._data_processor.transform_data(Xsplit), pred_leaf=True)
                new_probs = onehot_sparse(probs)
                actual_probs = self._classifier2.predict_proba(new_probs)[:,1]
                results = np.append(results, actual_probs)
                del Xsplit
                del probs
                del new_probs
                del actual_probs

            print('results.shape: {}'.format(results.shape))
            print('resutls.transposed.shape: {}'.format(results.T.shape))
            return results
        return []

    def _convert_nan_to_num(self, F, info_dict):
        # Convert time and numerical nan
        data = F['numerical']
        data = np.nan_to_num(data)

        # Convert categorical nan
        if info_dict['no_of_categorical_features'] > 0:
            categorical_data = F['CAT'].fillna('nan').values
            data = np.concatenate((data, categorical_data), axis=1)
            del categorical_data

        # Convert mvc nan
        if info_dict['no_of_mvc_features'] > 0:
            mvc_data = F['MV'].fillna('nan').values
            data = np.concatenate((data, mvc_data), axis=1)
            del mvc_data
        return data

    def _has_sufficient_time(self, info_dict):
        return info_dict['dataset_time_spent'] < info_dict['time_budget'] * self._dataset_budget_threshold

    def _too_much_training_data(self):
        return self._train_data.size > self._max_train_data

    def _find_best_hyperparameters(self):

        param_choice_fixed = { 
            'n_estimators':400, 
            'learning_rate':0.01, 
            'num_leaves':50, 
            'feature_fraction':0.6, 
            'bagging_fraction':0.6, 
            'bagging_freq':2, 
            'boosting_type':'gbdt', 
            'objective':'binary', 
            'metric':'auc' 
        }
        
        autohyper = HyperparametersTuner(parameter_space=param_choice_fixed)
        best_score_choice1 = autohyper.fit(self._transformed_train_data, self._transformed_train_labels.ravel(), 1)
    
        #Get the AUC for the fixed hyperparameter+Hyperopt combination on the internal validation set
        #Step:1-Define the search space for Hyperopt to be a small delta region over the initial set of fixed hyperparameters 
        n_estimators_low = param_choice_fixed['n_estimators'] - self.delta_n_estimators
        n_estimators_high = param_choice_fixed['n_estimators'] + self.delta_n_estimators
        
        learning_rate_low = np.log(0.001) if (param_choice_fixed['learning_rate'] - self.delta_learning_rate)<0.001 else np.log(param_choice_fixed['learning_rate'] - self.delta_learning_rate)
        learning_rate_high = np.log(param_choice_fixed['learning_rate'] + self.delta_learning_rate)
        
        num_leaves_low = param_choice_fixed['num_leaves'] - self.delta_num_leaves
        num_leaves_high = param_choice_fixed['num_leaves'] + self.delta_num_leaves
        
        feature_fraction_low = np.log(0.05) if (param_choice_fixed['feature_fraction'] - self.delta_feature_fraction)<0.05 else np.log(param_choice_fixed['feature_fraction'] - self.delta_feature_fraction)
        feature_fraction_high = np.log(1.0) if (param_choice_fixed['feature_fraction'] + self.delta_feature_fraction)>1.0 else np.log(param_choice_fixed['feature_fraction'] + self.delta_feature_fraction)
        
        bagging_fraction_low = np.log(0.05) if (param_choice_fixed['bagging_fraction'] - self.delta_bagging_fraction)<0.05 else np.log(param_choice_fixed['bagging_fraction'] - self.delta_bagging_fraction)
        bagging_fraction_high = np.log(1.0) if (param_choice_fixed['bagging_fraction'] + self.delta_bagging_fraction)>1.0 else np.log(param_choice_fixed['bagging_fraction'] + self.delta_bagging_fraction)
        
        bagging_freq_low = 1 if (param_choice_fixed['bagging_freq'] - self.delta_bagging_freq)<1 else param_choice_fixed['bagging_freq'] - self.delta_bagging_freq
        bagging_freq_high = param_choice_fixed['bagging_freq'] + self.delta_bagging_freq
        
        boosting_type = param_choice_fixed['boosting_type']
        objective = param_choice_fixed['objective']
        metric = param_choice_fixed['metric']

        #set the search space to be explored by Hyperopt
        param_space_forFixed = {
            'objective': "binary",
            'n_estimators' : hp.choice('n_estimators', np.arange(n_estimators_low, n_estimators_high, 50, dtype=int)),
            'num_leaves': hp.choice('num_leaves',np.arange(num_leaves_low, num_leaves_high, 5, dtype=int)),
            'feature_fraction': hp.loguniform('feature_fraction', feature_fraction_low, feature_fraction_high),
            'bagging_fraction': hp.loguniform('bagging_fraction', bagging_fraction_low, bagging_fraction_high), 
            'bagging_freq': hp.choice ('bagging_freq',np.arange(bagging_freq_low, bagging_freq_high+1, 1, dtype=int)),
            'learning_rate': hp.loguniform('learning_rate', learning_rate_low, learning_rate_high), 
            'boosting_type' : boosting_type,
            'metric': metric,
            'verbose':-1
        }
            
        #run Hyperopt to search nearby region in the hope to obtain a better combination of hyper-parameters
        autohyper = HyperparametersTuner(max_evaluations=self.max_evaluation, parameter_space=param_space_forFixed) 
        best_hyperparams_choice2, best_score_choice2 = autohyper.fit(self._transformed_train_data, self._transformed_train_labels.ravel(), 0)
        
        #Compare choice-1 & choice-2 and take the better one
        if best_score_choice1 >= best_score_choice2:
            self.best_hyperparams = param_choice_fixed
        else:
            self.best_hyperparams = best_hyperparams_choice2
        
        print('\nBest Hyperparams: {}\n'.format(self.best_hyperparams))