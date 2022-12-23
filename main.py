from utils.Optimizer import Optimizer

import logging
import pandas as pd


if __name__ == '__main__':
    """
    This is the main function of the script used for debugging
    """
    # Instanciate a logging with debug level
    logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)
    # load the data
    data = pd.read_csv('data/adult.csv')
    data.drop(['fnlwgt','education','occupation','relationship','native-country'], axis=1, inplace=True)
    logging.info(data.columns)
    logging.info(data.dtypes)
    # transform target to binary
    #data['income'] = data['income'].apply(lambda x: 0 if x == ' <=50K' else 1)
    # apply one hot encoding
    data = pd.get_dummies(data,drop_first=True)
    # instanciate the class
    opt = Optimizer(model_type='randomforest',
                    data=data,
                    target= 'age',
                    seed=42,
                    max_evals=3,
                    cv_splits=3
                    )
    # optimize the search space
    best_params = opt.optimize()
    logging.info(f" Best Parameters: {opt.best_parameters}")
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