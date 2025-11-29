
import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy

circRNA_disease_M = np.loadtxt(r"association.txt", dtype = int)

#计算cicrRNA拉普拉斯核相似性
def Laplace_circRNA():
    row = circRNA_disease_M.shape[0]
    a = 2

    SM1 = np.zeros((row, row))

    for i in range(0, row):
        for j in range(0, row):
            SM1[i,j] = math.exp(-(1/a)*np.linalg.norm((circRNA_disease_M[i,]-circRNA_disease_M[j,])))

    
    GSM = SM1
    return GSM

#计算疾病拉普拉斯核相似性
def Laplace_disease():
    column = circRNA_disease_M.shape[1]
    a = 2

    lap_disease_sim = np.zeros((column, column))

    for i in range(0, column):
        for j in range(0, column):
            lap_disease_sim[i,j] = math.exp(-(1/a)*np.linalg.norm((circRNA_disease_M[:,i]-circRNA_disease_M[:,j])))

    GmiRNA = lap_disease_sim
    return GmiRNA

def main():
    LKS_circRNA = Laplace_circRNA()
    LKS_disease = Laplace_disease()
    print(LKS_disease, LKS_disease.shape)

    np.savetxt(r'LKS_circRNA.txt', LKS_circRNA, delimiter='\t', fmt = '%.9f')
    np.savetxt(r'LKS_disease.txt', LKS_disease, delimiter='\t', fmt='%.9f')


if __name__ == "__main__":
    main()