import pip
import time

def pip_install(package):
    pip.main(['install', package])

def extract(data_info, time_info):
    time_budget = data_info['time_budget']
    no_of_time_features = data_info['loaded_feat_types'][0]
    no_of_numerical_features = data_info['loaded_feat_types'][1]
    no_of_categorical_features = data_info['loaded_feat_types'][2]
    no_of_mvc_features = data_info['loaded_feat_types'][3]
    total_no_of_features = no_of_time_features + no_of_numerical_features + \
                           no_of_categorical_features + no_of_mvc_features

    time_starting_index = 0
    numerical_starting_index = no_of_time_features
    categorical_starting_index = numerical_starting_index + no_of_numerical_features
    mvc_starting_index = categorical_starting_index + no_of_categorical_features

    current_time = time.time() 
    overall_time_spent = current_time - time_info[0]
    dataset_time_spent = current_time- time_info[1]

    return {
        'time_budget': time_budget,
        'no_of_time_features': no_of_time_features,
        'no_of_numerical_features': no_of_numerical_features,
        'no_of_categorical_features': no_of_categorical_features,
        'no_of_mvc_features': no_of_mvc_features,
        'total_no_of_features': total_no_of_features,
        'time_starting_index': time_starting_index,
        'numerical_starting_index': numerical_starting_index,
        'categorical_starting_index': categorical_starting_index,
        'mvc_starting_index': mvc_starting_index,
        'overall_time_spent': overall_time_spent,
        'dataset_time_spent': dataset_time_spent
    }

def print_data_info(info_dict):
    print('Dataset budget: {0:d} seconds'.format(info_dict['time_budget']))
    print('No. of time features: {0:d}'.format(info_dict['no_of_time_features']))
    print('No. of numerical features: {0:d}'.format(info_dict['no_of_numerical_features']))
    print('No. of categorical features: {0:d}'.format(info_dict['no_of_categorical_features']))
    print('No. of mvc features: {0:d}'.format(info_dict['no_of_mvc_features']))

def print_time_info(info_dict):
    print('Overall time spent: {0:5.2f} seconds'.format(info_dict['overall_time_spent']))
    print('Dataset time spent: {0:5.2f} seconds'.format(info_dict['dataset_time_spent'])) 