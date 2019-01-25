import pandas as pd
import numpy as np
from collections import Counter

class DataProcessor:

    def __init__(self, info_dict):
        self.is_uninitialized = True

        self._time_col_indices = np.arange(info_dict['time_starting_index'], \
                                           info_dict['numerical_starting_index'])

        self._numerical_col_indices = np.arange(info_dict['numerical_starting_index'], \
                                                info_dict['categorical_starting_index'])

        self._categorical_col_indices = np.arange(info_dict['categorical_starting_index'], \
                                                  info_dict['mvc_starting_index'])

        self._mvc_col_indices = np.arange(info_dict['mvc_starting_index'], \
                                          info_dict['total_no_of_features'])

        self._feature_map = {}
        for col_index in np.concatenate((self._categorical_col_indices, self._mvc_col_indices)):
            self._feature_map[col_index] = {}

        self._time_map = {}
        for col_index in self._time_col_indices:
            self._time_map[col_index] = 0.0

        self._frequency_map = {}

    def preprocess(self, data):
        for col_index in range(data.shape[1]):
            if self._is_time_col(col_index):
                self._get_min_time(data, col_index)
            elif self._is_categorical_col(col_index) or self._is_mvc_col(col_index):
                self._count_frequency(data, col_index)

        self.is_uninitialized = False

    def prepare_frequency_map(self):
        for col_index in np.concatenate((self._categorical_col_indices, self._mvc_col_indices)):
            keys = self._feature_map[col_index].keys()
            vals = np.array(list(self._feature_map[col_index].values())).astype(float)
            self._frequency_map[col_index] = dict(zip(keys,vals))

    def transform_data(self, data):
        result = []
        for col_index in range(data.shape[1]):
            if self._is_categorical_col(col_index) or self._is_mvc_col(col_index):
                freq_encoded_col = np.vectorize(self._frequency_map[col_index].get)(data[:,col_index])
                result.append(freq_encoded_col)

            elif self._is_time_col(col_index):
                transformed_date_col = data[:,col_index].astype(float) - self._time_map[col_index]
                result.append(transformed_date_col)
            
            elif self._is_numerical_col(col_index):
                result.append(data[:,col_index])

        for i in range(len(self._time_col_indices)):
            for j in range(i+1, len(self._time_col_indices)):
                if len(np.nonzero(data[:,i])) > 0 and len(np.nonzero(data[:,j])) > 0:
                    result.append(data[:,i] - data[:,j])

            dates = pd.DatetimeIndex(data[:,i])
            dayofweek = dates.dayofweek.values
            dayofyear = dates.dayofyear.values
            month = dates.month.values
            weekofyear = dates.weekofyear.values
            day = dates.day.values
            hour = dates.hour.values
            minute = dates.minute.values
            year = dates.year.values

            result.append(dayofweek)
            result.append(dayofyear)
            result.append(month)
            result.append(weekofyear)
            result.append(year)
            result.append(day)
            result.append(hour)
            result.append(minute)

        return np.array(result).T

    def _get_min_time(self, data, col_index):
        date_col = data[:,col_index].astype(float)
        non_zero_indices = np.nonzero(date_col)[0]

        if non_zero_indices.size != 0:
            if self._time_map[col_index] == 0:
                self._time_map[col_index] = np.min(date_col[non_zero_indices])
            else:
                self._time_map[col_index] = np.min([self._time_map[col_index], \
                                                    np.min(date_col[non_zero_indices])])

    def _count_frequency(self, data, col_index):
        curr_feature_map = dict(pd.value_counts(data[:,col_index]))
        self._feature_map[col_index] = dict(Counter(self._feature_map[col_index]) + \
                                            Counter(curr_feature_map))

    def _is_time_col(self, col_index):
        return col_index in self._time_col_indices
    
    def _is_numerical_col(self, col_index):
        return col_index in self._numerical_col_indices

    def _is_categorical_col(self, col_index):
        return col_index in self._categorical_col_indices

    def _is_mvc_col(self, col_index):
        return col_index in self._mvc_col_indices

        