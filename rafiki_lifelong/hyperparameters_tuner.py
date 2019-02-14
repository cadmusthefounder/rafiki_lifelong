from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
from hyperopt import space_eval
from lightgbm import LGBMClassifier
from libscores import *

class HyperparametersTuner:

    def __init__(self,max_evaluations=25,seed=1,parameter_space={}):
        self.max_evaluations = max_evaluations
        self.test_size = 0.25 ## fraction of data used for internal validation
        self.shuffle = False
        self.best_params = {}
        self.seed = seed
        self.param_space = parameter_space
  
    def gbc_objective(self,space):

        model = LGBMClassifier(random_state=self.seed,min_data=1, min_data_in_bin=1)
        model.set_params(**space)
        model.fit(self.Xe_train,self.ys_train)
        mypreds = model.predict_proba(self.Xe_test)[:,1]
        auc = auc_metric(self.ys_test.reshape(-1,1),mypreds.reshape(-1,1))
        return{'loss': (1-auc), 'status': STATUS_OK }

    def fit(self,X,y,indicator): 
        '''
        indicator=1 means we intend to do just sampling and one-time fitting
        for evaluating a fixed set of hyper-parameters, 
        0 means run hyperopt to search in the neighborhood of the seed 
        hyper-parameters to see if model quality is improving.
        '''

        XFull = X
        yFull = y
        self.Xe_train, self.Xe_test, self.ys_train, self.ys_test = \
        train_test_split(XFull, yFull.ravel(),test_size = self.test_size, random_state=self.seed,shuffle=True)
 
        if indicator == 1: 
            ## just fit lightgbm once to obtain the AUC w.r.t a fixed set of hyper-parameters ##
            model = LGBMClassifier(random_state=self.seed,min_data=1, min_data_in_bin=1)
            model.set_params(**self.param_space) 
            model.fit(self.Xe_train,self.ys_train)
            mypreds = model.predict_proba(self.Xe_test)[:,1]
            auc = auc_metric(self.ys_test.reshape(-1,1),mypreds.reshape(-1,1))
            return auc
        else:
            trials = Trials()
            best = fmin(fn=self.gbc_objective,space=self.param_space,algo=tpe.suggest,trials=trials,max_evals=self.max_evaluations)
            params = space_eval(self.param_space, best)
            self.best_params = params
            return params, 1-np.min([x['loss'] for x in trials.results])
