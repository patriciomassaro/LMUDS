# Data Ethics & Security @ LMU

This repository contains the course material for the course "Data Ethics & Security" at LMU Munich.

It consists mainly on the implementation pipeline explained in the article and a notebook to plot it's results.

## Initialization

- Clone the repository
- install requirements with
 `pip install -r requirements.txt`

To run the pipeline, just run main.py. To see the results, go to the results_analysis notebook in the notebooks folder.

In case any error happens, after running main.py you will see a log : app.log that may help you to understand the error.

## Elements

- utils: Contains the modules used in the pipeline
    - Anonymization: Contains the anonymization algorithms.
    - Preprocessing.py: Contains the preprocessing functions.
    - Optimizer.py: Contains the optimizer class that executes the ML tasks.
- main.py: main file to run the pipeline ( Note: It may take some time)
- metrics: contains the metrics obtain through pipeline runs
- results: folder with the anonymized & preprocessed datasets
- notebook: Notebooks used to analyze the results


