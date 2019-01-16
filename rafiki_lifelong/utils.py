import pip
import time

def pip_install(module):
    pip.main(['install', package])

def extract(data_info, time_info):
    time_budget = datainfo['time_budget']
    no_of_time_features = datainfo['loaded_feat_types'][0]
    no_of_numerical_features = datainfo['loaded_feat_types'][1]
    no_of_categorical_features = datainfo['loaded_feat_types'][2]
    no_of_mvc_features = datainfo['loaded_feat_types'][3]

    current_time = time.time() 
    overall_time_spent = current_time - timeinfo[0]
    dataset_time_spent = current_time- timeinfo[1]

    return {
        'time_budget': time_budget,
        'no_of_time_features': no_of_time_features,
        'no_of_numerical_features': no_of_numerical_features,
        'no_of_categorical_features': no_of_categorical_features,
        'no_of_mvc_features': no_of_mvc_features,
        'overall_time_spent': overall_spent_time,
        'dataset_time_spent': dataset_spent_time
    }
