import numpy as np
import random

class Sampler:

    def __init__(self):
        pass
    
    def majority_undersampling(self, data, labels, fraction=3.0, seed=1):
        labels = labels.reshape(len(labels))
        class_0_freq = len(labels[labels == 0])
        class_1_freq = len(labels[labels == 1])
        majority_class = 0
        
        if class_1_freq > class_0_freq:
            majority_class = 1
            minority_count = class_0_freq
        else:
            minority_count = class_1_freq

        minority_class = int(not majority_class)
        indices = np.array(range(len(labels)))
        majority_ind = indices[labels == majority_class]
        minority_index = indices[labels == minority_class]
        np.random.seed(seed)
        if int(minority_count * fraction) > len(majority_ind):
            size = len(majority_ind)
        else:
            size = int(minority_count * fraction)
        majority_index = np.random.choice(indices[labels == majority_class], size=size, replace=False)
        sorted_index = sorted(np.concatenate([minority_index, majority_index]))

        return data[sorted_index], labels[sorted_index]

    def random_sample_in_order(self, data, labels, remove_percentage=0, seed=1):
        if remove_percentage == 0:
            return data, labels

        num_train_samples = len(data)
        rem_samples = int(num_train_samples * remove_percentage)
        # random.seed(seed)
        skip = sorted(random.sample(range(num_train_samples), num_train_samples - rem_samples))
        return data[skip,:], labels[skip,:]