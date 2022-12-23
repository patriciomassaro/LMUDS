from hyperopt import hp,Trials,STATUS_OK,fmin,tpe
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score,mean_squared_error,mean_absolute_error,classification_report

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
            'n_jobs': -1,
}

class Optimizer:
    """
    This class is used to optimize the hyperparameters of a model using hyperopt

    Parameters
    ----------
    model_type: str ( 'xgboost' or 'randomforest') 
        The model type to optimize
    data: pd.DataFrame
        The data to use for optimization, it should be already preprocessed
    target: str
        The target column name
    max_evals: int
        The maximum number of evaluations to perform by hyperopt
    cv_splits: int
        The number of cross validation splits to perform
    seed: int
        The seed to use for reproducibility
    """
    def __init__(self, model_type:str,data:pd.DataFrame,seed:int, target:str, max_evals:int=10, cv_splits:int=5):
        self.model = model_type
        self.max_evals = max_evals
        self.trials = Trials()
        self.best = None
        self.splits = cv_splits
        self.seed = seed
        self.target = target
        self.data = data
        

    def define_problem_type(self):
        """
        This function is used to define the problem type based on the unique values of the target
        """
        if len(self.train_target.unique()) > 2:
            self.problem_type = 'regression'
        else:
            self.problem_type = 'binary'

    def split_into_train_and_test(self):
        train,test = train_test_split(self.data,test_size=0.2,random_state=self.seed)
        self.train_features = (train.drop(self.target,axis=1))
        self.train_target = train[self.target]
        self.test_features = (test.drop(self.target,axis=1))
        self.test_target = test[self.target]

        

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
            else:
                self.search_space['objective'] = 'binary:logistic' 
                self.search_space['eval_metric'] = 'logloss'
                self.metric='logloss'
        else:
            self.search_space = RF_SEARCH_SPACE

                

    def function_to_optimize(self, params):
        """
        This function is used to optimize the search space based on the model type and problem type
        """
        if self.model == 'xgboost':
            # convert data and target to DMatrix
            data = xgb.DMatrix(self.train_features, self.train_target)
            # perform a cross validation with the given parameters and return the  mean evaluation metric 
            cv_results = xgb.cv(params, data, nfold=self.splits,num_boost_round=100)
            print(cv_results)
            return {'status':STATUS_OK, 'loss':cv_results[f'test-{self.metric}-mean'].iloc[-1], 'attributes':params}
        elif self.model == 'randomforest':
            # perform a cross validation with the given parameters and return the  mean evaluation metric 
            if self.problem_type == 'regression':
                cv_results = cross_val_score(RandomForestRegressor(**params), self.train_features, self.train_target, cv=self.splits,error_score='raise')
            else:
                cv_results = cross_val_score(RandomForestClassifier(**params), self.train_features, self.train_target, cv=self.splits,error_score='raise',scoring="neg_log_loss")
                # We want to minimize the log loss, so we multiply by -1
                cv_results = -cv_results
                print(cv_results)
                
            return {'status':STATUS_OK, 'loss':np.mean(cv_results), 'attributes':params}

            
            
    def optimize(self):
        """
         Perform the optimization of the search space using hyperopt fmin function
        """
        #split the data into train and test
        self.split_into_train_and_test()
        # determine the problem type
        self.define_problem_type()
        # complete the search space
        self.complete_search_space()
        # optimize the search space
        self.best_parameters = fmin(self.function_to_optimize, self.search_space, algo=tpe.suggest,
                         max_evals=self.max_evals, trials=self.trials)
        
        # return the best parameters
        return self.best_parameters

    
    def train_best_model(self):
        """
        This function is used to train the best model found during the optimization
        """
        # train the model with the best parameters found during the optimization
        if self.model == 'xgboost':
            # convert parameters to integers
            self.best_parameters['max_depth'] = int(self.best_parameters['max_depth'])
            # Train the model            
            self.best_model = xgb.train(self.best_parameters, xgb.DMatrix(self.train_features, self.train_target),num_boost_round=100)
        elif self.model == 'randomforest':
            # Convert int parameters to integers
            self.best_parameters['n_estimators'] = int(self.best_parameters['n_estimators'])
            self.best_parameters['max_depth'] = int(self.best_parameters['max_depth'])
            self.best_parameters['min_samples_split'] = int(self.best_parameters['min_samples_split'])
            self.best_parameters['min_samples_leaf'] = int(self.best_parameters['min_samples_leaf'])
            
            # Set regressor or classifier
            if self.problem_type == 'binary':
                self.best_model = RandomForestClassifier(**self.best_parameters)
            else:
                self.best_model = RandomForestRegressor(**self.best_parameters)
            
            # Train the model
            self.best_model.fit(self.train_features, self.train_target)            
        
        # return the best model
        return self.best_model

    def make_predictions_from_best_model(self):
        """
        This function is used to check the performance of the best model found during the optimization on the test set
        """
        # predict labels and probabilities in case the problem is binary
        if self.problem_type == 'binary':
            if self.model == 'xgboost':
                self.test_proba_predictions = self.best_model.predict(xgb.DMatrix(self.test_features))
                self.test_label_predictions = np.where(self.test_proba_predictions > 0.5, 1, 0)
            elif self.model == 'randomforest':
                self.test_proba_predictions = self.best_model.predict_proba(self.test_features)[:,1]
                self.test_label_predictions = self.best_model.predict(self.test_features)
        # predict labels in case the problem is regression
        elif self.problem_type == 'regression':
            self.test_label_predictions = self.best_model.predict(xgb.DMatrix(self.test_features))

    def report_metrics(self):
        """
        This function is used to report the performance of the best model found during the optimization on the test set
        """
        
        # report the performance of the best model found during the optimization on the test set
        if self.problem_type == 'binary':
            print(classification_report(self.test_target, self.test_label_predictions))
            print(roc_auc_score(self.test_target, self.test_proba_predictions))
        elif self.problem_type == 'regression':
            print(np.sqrt(mean_squared_error(self.test_target, self.test_label_predictions)))


        

        


if __name__ == '__main__':
    """
    This is the main function of the script used for debugging
    """
    # load the data
    data = pd.read_csv('data/adult.csv')
    data.drop(['fnlwgt','education','occupation','relationship','native-country'], axis=1, inplace=True)
    print(data.columns)
    print(data.dtypes)
    # transform target to binary
    #data['income'] = data['income'].apply(lambda x: 0 if x == ' <=50K' else 1)
    # apply one hot encoding
    data = pd.get_dummies(data,drop_first=True)
    # instanciate the class
    opt = Optimizer(model_type='xgboost',
                    data=data,
                    target= 'age',
                    seed=42,
                    max_evals=30,
                    cv_splits=3
                    )
    # optimize the search space
    best_params = opt.optimize()
    print(opt.best_parameters)
    # train the best model
    best_model = opt.train_best_model()
    # make predictions from the best model in the test set
    opt.make_predictions_from_best_model()
    # report the performance of the best model in the test set
    opt.report_metrics()




# Main loop
# Datasets [ A10k, A100k, A1M]
# K [1,10,50,100]
# Anonimization algorithm [k-anonymity, l-diversity, t-closeness]
# Targets = [A10k, A100k, A1M]
# For anonimization algorithm
    # for datasaet in datasets 
        # Preprocess dataset
        # Split
        # for k in K
            # apply anonimization ( parameter k )
            # Optimize with train dataset
            # Save metrics for training dataset
            # Save metrics for testing dataset
            # Repeat 
        #Repeat


# plot