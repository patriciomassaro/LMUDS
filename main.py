from utils.anonymization.anonymize import Anonymizer
from utils.Preprocessing import preprocess_data
from utils.Optimizer import Optimizer

import pandas as pd
import logging
import json
import os
import glob


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

    # This 2 lines must change, choose 2 or 3 DS and Method
    datasets = ["adult",
                "cahousing", 
                "cmc"
            ]
    methods = [#"mondrian",
               #"topdown",
               #"cluster",
               #"mondrian_ldiv",
               "classic_mondrian",
               #"datafly"
               ]
    ks = [10,350,400]

    # Anonymization
    """
    Path to the anonymized datasets:  Results/%DATASET%/%METHOD%/%DATASET%_anonymized_%K%.csv
    """
    for dataset in datasets:
        for method in methods:
            for k in ks:

                file_name = dataset+"_anonymized_"+str(k)+".csv"

                if os.path.exists("results/"+dataset+"/"+method+"/"+file_name):
                    logging.info(file_name + " already exists!")
                else:
                    logging.info("Creating " + file_name)
                    args = {'method': method, 'k': k, 'dataset': dataset}
                    anonymizer = Anonymizer(args)
                    anonymizer.anonymize()




    # Preprocessing loop
    """
    We take the anonymized datasets, apply preprocessing functions. Result : A folder with datasets ready to be used by the optimizer

    Path to the anonymized preprocessed datasets:  Results/%DATASET%/%METHOD%/%DATASET%_anonymized_%K%_prep.csv
    """

    # Get all the path of the anonymized datasets using blob
    anonymized_datasets =  [fn for fn in glob.glob("results/**/**/*_anonymized_*.csv") if not 'prep' in fn]

    for anonymized_dataset_path in anonymized_datasets:
        preprocessed_dataset_path = anonymized_dataset_path.replace('.csv','_prep.csv')
        logging.info(f"Preprocessing {anonymized_dataset_path}")
        # read the dataset and preprocess it
        if os.path.exists(preprocessed_dataset_path):
                    logging.info( preprocessed_dataset_path + " already exists!")
        else:
            logging.info("Creating " + preprocessed_dataset_path)
            data = pd.read_csv(anonymized_dataset_path, sep=';', header=0, encoding='ascii')
            preprocessed_dataset = preprocess_data(data)
            # Save the preprocessed datasets in the same folder
            preprocessed_dataset.to_csv(preprocessed_dataset_path, index=False, sep=';')
            logging.info(f"Saving to {preprocessed_dataset_path}")
        

    ### TODO: Read each anonymized preprocessed dataset, apply the optimizer and save the metrics in a json file.
    
    ml_algorithms = ['xgboost','randomforest']

    # Dictionary to save the metrics
    metrics = {}
    
    for dataset in datasets:
        # Create a dictionary for each dataset
        metrics[dataset]={}
        # load dataset and preprocess it for each target they have

        # FOR TESTING PURPOSE, CHANGE
        data = pd.read_csv(f'results/adult/mondrian/adult_anonymized_2.csv', sep=';', header=0, encoding='ascii')
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

