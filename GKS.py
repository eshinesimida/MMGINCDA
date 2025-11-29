#高斯核相似性方法
import numpy as np
import pandas as pd
import math
#import numpy.matlib
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy
from scipy.spatial.distance import cdist
import os
print(os.getcwd())

circRNA_disease_M = np.loadtxt(r"association1.txt", dtype = int)
print(circRNA_disease_M.shape, np.sum(circRNA_disease_M==1))


#计算circRNA高斯轮廓核相似性
def Gaussian_circRNA():
    row = circRNA_disease_M.shape[0]
    sum = 0
    CC1 = np.zeros((row,row))

    row_norms = np.linalg.norm(circRNA_disease_M, axis = 1) #所有行范数
    sum = np.sum(row_norms**2) #直接求和

    ps = row/sum

    distances = cdist(circRNA_disease_M, circRNA_disease_M, 'euclidean')
    CC1 = np.exp(-ps * distances**2)

    CC = CC1
    return CC
    #print(CC1, CC1.shape)

#计算疾病高斯轮廓核相似性
def Gaussian_disease():
    column = circRNA_disease_M.shape[1]
    sum = 0
    DD1 = np.zeros((column,column))

    row_cols = np.linalg.norm(circRNA_disease_M, axis = 0) #所有列范数
    sum = np.sum(row_cols**2) #直接求和

    ps = column/sum

    distances = cdist(circRNA_disease_M.T, circRNA_disease_M.T, 'euclidean')
    DD1 = np.exp(-ps * distances**2)

    DD = DD1
    return DD


def main():
    GKS_circRNA = Gaussian_circRNA()
    GKS_disease = Gaussian_disease()
    #print(GKS_circRNA, GKS_circRNA.shape)
    #print(GKS_disease, GKS_disease.shape)
    np.savetxt(r'GKS_circRNA-D1.txt', GKS_circRNA, delimiter='\t', fmt='%.9f')
    np.savetxt(r'GKS_disease-D1.txt', GKS_disease, delimiter='\t', fmt='%.9f')

if __name__ == "__main__":
    main()
