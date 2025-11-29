import numpy as np

def compute_GS_matrix(M_bar, alpha=0.5, eps = 1e-06):
    n = M_bar.shape[0]
    #构建单位矩阵
    I = np.eye(n)

    #对列进行归一化
    column_sums = M_bar.sum(axis=0)
    M_bar_normalized = M_bar/column_sums

    #计算逆矩阵
    inv_matrix = np.linalg.inv(I-alpha*M_bar_normalized)

    #初始化全局相似性
    GS_matrix = np.zeros((n,n))

    #计算每个全局相似性
    # for i in range(n):
    #     m = np.zeros((n,1))
    #     m[i] = 1
    #
    #     #
    #     m_tilde = (1-alpha)*inv_matrix.dot(m)
    #
    #     #结果存到全局相似性矩阵中
    #     GS_matrix[:, i]= m_tilde.flatten()
    GS_matrix = (1-alpha)*inv_matrix

    return GS_matrix



