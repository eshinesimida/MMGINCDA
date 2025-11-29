# MMGINCDA

We propose a new computational model, Multiple similarity and multiple kernel fusion based on graph inference network for predicting circRNA-disease associations. This is the implementation of MMGINCDA:



# Environment Requirement

    python == 3.9.13
    torch == 2.5.1
    torch-genometric == 1.4.2
    matplotlib == 3.5.2
    networkx == 2.8.4
    numpy == 1.21.6
    pandas == 1.4.2
    scipy == 1.9.1
    
# Dataset

We performed 5-fold cross validation on four datasets. Dataset1-4 are from CircR2Disease database, CircRNADisease database, Circ2Disease database, and CircR2Disease v2.0, respectively. We divided the known circRNA-disease associations into five equal parts and stored them in .txt files.

# Model

    ## Main.py: This file contains the main function. The paramaters of LGCDA are also adjusted in this file.
    models.py: This file contains model building.
    preprocessing.py: This file contains data reading.
    similarty_calculated.py: This file records the detail of computing Gaussian similarities and cosine similarities.
    sortscore.py: The prediction score of each circRNA-disease pair is sorted in this file before computing the AUC/AUPR.
    train_eval.py: This file records the detail of model training.
    util_functions.py: Constructing closed local subgraphs for (circRNA, disease) association pairs in this file.
