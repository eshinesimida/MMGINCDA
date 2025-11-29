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

    MMGINCDA.py: This file contains the main function. The paramaters of MMGINCDA are also adjusted in this file.
    GKS.py: This file contains calculating the Gaussian kernel similarity.
    Global_similarity.py: This file contains calculating the global similarity.
    LKS1.py: This file contains calculating the Laplace kernel similarity.
    Local_similarity.py: This file contains calculating the local similarity.
    SKF1.py: This file records the process of model fusion.
    known.py: This file contains the knonw and unkonw circRNA-disease associations.
