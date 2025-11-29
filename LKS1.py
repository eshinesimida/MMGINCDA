import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy
from scipy.spatial.distance import cdist

circRNA_disease_M = np.loadtxt(r"association1.txt", dtype=int)
print(circRNA_disease_M.shape, np.sum(circRNA_disease_M==1))

# 计算cicrRNA拉普拉斯核相似性
def Laplace_circRNA():
    row = circRNA_disease_M.shape[0]
    a = 2

    SM1 = np.zeros((row, row))

    distances = cdist(circRNA_disease_M, circRNA_disease_M, 'euclidean')
    SM1 = np.exp(-(1/a) * distances)
    GSM = SM1
    return GSM


# 计算疾病拉普拉斯核相似性
def Laplace_disease():
    column = circRNA_disease_M.shape[1]
    a = 2

    lap_disease_sim = np.zeros((column, column))

    distances = cdist(circRNA_disease_M.T, circRNA_disease_M.T, 'euclidean')
    lap_disease_sim = np.exp(-(1 / a) * distances)
    GmiRNA = lap_disease_sim
    return GmiRNA


def main():
    LKS_circRNA = Laplace_circRNA()
    LKS_disease = Laplace_disease()
    print(LKS_disease, LKS_disease.shape)

    np.savetxt(r'LKS_circRNA-D1.txt', LKS_circRNA, delimiter='\t', fmt='%.9f')
    np.savetxt(r'LKS_disease-D1.txt', LKS_disease, delimiter='\t', fmt='%.9f')


if __name__ == "__main__":
    main()