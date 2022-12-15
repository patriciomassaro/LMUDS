from hyperopt import hp,Trials,STATUS_OK,fmin,tpe
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import pandas as pd
import numpy as np

XGBOOST_SEARCH_SPACE = {
            'learning_rate': hp.loguniform('learning_rate', -4, -1),
            'max_depth': scope.int(hp.uniform('max_depth', 3, 10)),
            'min_child_weight': hp.loguniform('min_child_weight', -2, 2),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'gamma': hp.loguniform('gamma', -2, 2),
            'alpha': hp.loguniform('alpha', -2, 2),
            'lambda': hp.loguniform('lambda', -2, 2),
        }

RF_SEARCH_SPACE = {
            'n_estimators': scope.int(hp.uniform('n_estimators', 100, 1000)),
            'max_depth': scope.int(hp.uniform('max_depth', 3, 10)),
            'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 10)),
            'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 1, 10)),
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': hp.choice('bootstrap', [True, False]),
}

class Optimizer:
    """
    This class is used to optimize the hyperparameters of a model using hyperopt

    Parameters
    ----------
    model_type: str ( 'xgboost' or 'randomforest') 
        The model type to optimize
    data: pd.DataFrame
        The data to use for optimization
    target: str
        The target column name
    max_evals: int
        The maximum number of evaluations to perform by hyperopt
    cv_splits: int
        The number of cross validation splits to perform
    """
    def __init__(self, model_type:str,data:pd.DataFrame, target:str, max_evals:int=100, cv_splits:int=5):
        self.model = model_type
        self.max_evals = max_evals
        self.trials = Trials()
        self.best = None
        self.splits = cv_splits
        # split the data into features and target, doing one hot encoding for categorical features
        self.features= pd.get_dummies(data.drop(target,axis=1), drop_first=True)
        self.target=data[target]

    def define_problem_type(self):
        """
        This function is used to define the problem type based on the unique values of the target
        """
        if len(self.target.unique()) > 10:
            self.problem_type = 'regression'
        elif len(self.target.unique()) > 2:
            self.problem_type = 'multiclass'
        else:
            self.problem_type = 'binary'
        

    def complete_search_space(self):
        """
        This function is used to complete the search space based on the model type and problem type
        """
        # add the objective function to the search space based on the model type and problem type
        if self.model == 'xgboost':
            self.search_space = XGBOOST_SEARCH_SPACE
            if self.problem_type == 'regression':
                self.search_space['objective'] = 'reg:squarederror'
                self.metric = 'rmse'
            elif self.problem_type == 'multiclass':
                self.search_space['objective'] = 'multi:softmax'
                self.search_space['num_class'] = len(self.target.unique())
                self.metric = 'mlogloss'
            else:
                self.search_space['objective'] = 'binary:logistic'
                # use the rocauc metric for binary classification
                self.metric = 'auc'
                self.search_space['eval_metric'] = 'logloss'
                self.metric='logloss'
        elif self.model == 'randomforest':
            self.search_space = RF_SEARCH_SPACE
            if self.problem_type == 'regression':
                self.search_space['criterion'] = 'mse'
            else:
                self.search_space['criterion'] = 'gini'
                

    def function_to_optimize(self, params):
        """
        This function is used to optimize the search space based on the model type and problem type
        """
        if self.model == 'xgboost':
            # convert data and target to DMatrix
            data = xgb.DMatrix(self.features, self.target)
            # perform a cross validation with the given parameters and return the  mean evaluation metric 
            cv_results = xgb.cv(params, data, num_boost_round=100, nfold=self.splits)
            print(cv_results)
            return {'status':STATUS_OK, 'loss':cv_results[f'test-{self.metric}-mean'].iloc[-1], 'attributes':params}
        elif self.model == 'randomforest':
            # perform a cross validation with the given parameters and return the  mean evaluation metric 
            cv_results = cross_val_score(RandomForestRegressor(**params), self.features, self.target, cv=self.splits)
            return {'status':STATUS_OK, 'loss':np.mean(cv_results), 'attributes':params}

            
            
    def optimize(self):
        """
         Perform the optimization of the search space using hyperopt fmin function
        """
        # determine the problem type
        self.define_problem_type()
        # complete the search space
        self.complete_search_space()
        # optimize the search space
        self.best = fmin(self.function_to_optimize, self.search_space, algo=tpe.suggest,
                         max_evals=self.max_evals, trials=self.trials)
        
        # return the best parameters
        return self.best


if __name__ == '__main__':
    """
    This is the main function of the script used for debugging
    """
    # load the data
    data = pd.read_csv('data/adult.csv')
    data.drop(['fnlwgt','education','occupation','relationship','native-country'], axis=1, inplace=True)
    # transform target to binary
    data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    print(data.columns)
    print(data.dtypes)
    # instanciate the class
    opt = Optimizer('xgboost', data, 'income', max_evals=100)
    # optimize the search space
    best_params = opt.optimize()
    print(best_params)
