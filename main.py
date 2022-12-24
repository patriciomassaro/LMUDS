from utils.Optimizer import Optimizer
from utils.Preprocessing import preprocess_data

import logging
import pandas as pd
import json
import os


# 
def instanciate_log(logfilename:str = 'app.log'):
    # Remove the log file if it exists
    try:
        os.remove(logfilename)
    except OSError:
        pass
    # Instanciate a logging with debug level
    logging.basicConfig(filename=logfilename, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

if __name__ == '__main__':
    """
    This is the main function of the script 
    """
    METRICS_JSON_PATH = 'metrics.json'
    
    # Instanciate the log
    instanciate_log()

    datasets = ['adult.csv']
    ml_algorithms = ['xgboost','randomforest']
    anonymizations = ['Mondrian','SuperDuperAnonAlgo','t-closeness'] # Unused for now
    k = [1,10,50,100] # Unused for now


    # Dictionary to save the metrics
    metrics = {}
    
    for dataset in datasets:
        # Create a dictionary for each dataset
        metrics[dataset]={}
        # load dataset and preprocess it for each target they have
        data = pd.read_csv(f'data/{dataset}')
        preprocessed_datasets,targets = preprocess_data(data)

        for preprocessed_dataset in zip(preprocessed_datasets,targets):
            metrics[dataset][preprocessed_dataset[1]] = {}

            for ml_algorithm in ml_algorithms:
                # instanciate the Optimizer class class
                opt = Optimizer(model_type=ml_algorithm,
                                data=preprocessed_dataset[0],
                                target= preprocessed_dataset[1],
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
                # report the performance of the best model in the test set and save it in the dict
                metrics[dataset][preprocessed_dataset[1]][ml_algorithm] = opt.report_metrics()
    logging.info(f"Metrics: {metrics}")
    with open(f'data/metrics/{METRICS_JSON_PATH}', 'w') as f:
        json.dump(metrics, f)



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