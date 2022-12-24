import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


ADULT_TARGETS = ['income','age']

def determine_dataset(data):
    """
    Determine which dataset we are using based on the columns
    """
    if 'riskOfAccidentClass' in data.columns:
        return 'Insurance'
    elif 'capital-gain' in data.columns:
        return 'Adult'
    elif 'NativeCountry' in data.columns:
        return 'Irish'


def preprocess_adult(data:pd.DataFrame):
    """
    Preprocess the adult dataset, give one dataset per target
    """
    # Drop not important columns
    columns_to_drop = ['fnlwgt','education','native-country','capital-gain','capital-loss']
    logger.info (f'Dropping not important columns: {columns_to_drop}')
    data.drop(columns_to_drop, axis=1, inplace=True)

    preprocessed_datasets = []

    for target in ADULT_TARGETS:
        logger.info(f'Preprocessing data for target: {target}')
        preprocessed_data = data.copy()
        if target == 'income':
            # transform target to binary
            logger.info (f'Transforming income target to binary')
            preprocessed_data['income'] = preprocessed_data['income'].apply(lambda x: 0 if x == ' <=50K' else 1)

        # apply one hot encoding
        logger.info (f'Applying one hot encoding')
        preprocessed_data = pd.get_dummies(preprocessed_data,drop_first=True)
        logger.info(f'Columns after one hot encoding: {preprocessed_data.dtypes}')
        preprocessed_datasets.append(preprocessed_data)

    return preprocessed_datasets


def preprocess_data(data_raw:pd.DataFrame):
    """
    Preprocess the data
    """
    # Determine the dataset
    dataset = determine_dataset(data_raw)
    logger.info(f'Dataset to be used: {dataset}')
    # Preprocess the data
    if dataset == 'Adult':
        preprocessed_datasets = preprocess_adult(data_raw)
        targets = ADULT_TARGETS

    else:
        raise Exception('Dataset not supported')
    return preprocessed_datasets,targets
