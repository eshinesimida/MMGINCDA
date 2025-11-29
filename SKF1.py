import numpy as np

K1 = 5
K2 = 5

def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            row = [float(x) for x in line.split()]
            data.append(row)

    return np.array(data)

#列归一化
def column_normalize(matrix, eps=1e-6):
    normalized_matrix = np.zeros(matrix.shape)
    for j in range(matrix.shape[1]):
        column_sum = np.sum(matrix[:,j]) + eps
        normalized_matrix[:,j] = matrix[:, j]/column_sum
    return normalized_matrix

def calculate_neighbors(S, k):
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)

    return N


def row_normalization(S, k):
    # 计算邻居集合N
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]
        N[i] = list(neighbors)

    # 计算行归一化
    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        num = 0
        denominator = np.sum(S[i, N[i]])

        for j in range(S.shape[1]):
            if j in N[i]:
                if denominator != 0:
                    num = num + 1
                    result[i, j] = S[i, j] / denominator
                else:
                    result[i, j] = 0
            else:
                result[i, j] = 0
    return result

#load the data
#功能相似性
FS_circRNA = np.loadtxt('circFuncSimilarity.txt')
print(FS_circRNA, FS_circRNA.shape)
SS_disease = np.loadtxt('disease semantic similarity.txt')
print(SS_disease, SS_disease.shape)

#高斯核相似性
GKS_circRNA = np.loadtxt('GKS_circRNA-D1.txt')
print(GKS_circRNA, GKS_circRNA.shape)
GKS_disease = np.loadtxt('GKS_disease-D1.txt')
print(GKS_disease, GKS_disease.shape)

LKS_circRNA = np.loadtxt('LKS_circRNA-D1.txt')
LKS_disease = np.loadtxt('LKS_disease-D1.txt')
print(LKS_disease, LKS_disease.shape)
lty = np.loadtxt('association1.txt')
print(lty, lty.shape)
#
#calculate the neighbors N
N1 = calculate_neighbors(GKS_circRNA,K1)
print(N1[0],N1[1],N1[12])
N2 = calculate_neighbors(GKS_disease,K2)
N3 = calculate_neighbors(LKS_circRNA,K1)
N4 = calculate_neighbors(LKS_disease,K2)
N5 = calculate_neighbors(FS_circRNA,K1)
N6 = calculate_neighbors(SS_disease,K2)
print(N5)

#column normalization
GKS_circ_col = column_normalize(GKS_circRNA)
#print(GKS_circ_col, GKS_circ_col.shape)
GKS_dis_col = column_normalize(GKS_disease)
LKS_circ_col = column_normalize(LKS_circRNA)
LKS_dis_col = column_normalize(LKS_disease)
FS_circ_col = column_normalize(FS_circRNA)
SS_dis_col = column_normalize(SS_disease)
print(FS_circ_col,FS_circ_col.shape)
print(SS_dis_col,SS_dis_col.shape)

#
#row normaliztion
GKS_circ_row = row_normalization(GKS_circRNA, K1)
print(GKS_circ_row,GKS_circ_row.shape)
GKS_dis_row = row_normalization(GKS_disease, K2)
LKS_circ_row = row_normalization(LKS_circRNA, K1)
LKS_dis_row = row_normalization(LKS_disease, K2)
FS_circ_row = row_normalization(FS_circRNA,K1)
SS_dis_row = row_normalization(SS_disease,K2)
print(SS_dis_row, SS_dis_row.shape)
#
circRNA_P1 = GKS_circ_col
circRNA_P2 = LKS_circ_col
circRNA_P3 = FS_circ_col
circRNA_S1 = GKS_circ_row
circRNA_S2 = LKS_circ_row
circRNA_S3 = FS_circ_row
alpha_1 = 0.5

disease_P1 = GKS_dis_col
disease_P2 = LKS_dis_col
disease_P3 = SS_dis_col
disease_S1 = GKS_dis_row
disease_S2 = LKS_dis_row
disease_S3 = SS_dis_row

circRNA_P2_t = circRNA_P2
circRNA_P1_t = circRNA_P1
circRNA_P3_t = circRNA_P3
m=2
for i in range(1000):
    circRNA_p1 = alpha_1*(circRNA_S1@(circRNA_P2_t/2)@circRNA_S1.T)+\
                 (1-alpha_1)*(circRNA_P2/2)
    circRNA_p2 = alpha_1 * (circRNA_S2 @ (circRNA_P1_t / 2) @ circRNA_S2.T) + \
                 (1 - alpha_1) * (circRNA_P1 / 2)
    circRNA_p3 = alpha_1*(circRNA_S3 @ (circRNA_P1_t / 2) @ circRNA_S3.T) + \
                 (1 - alpha_1) * (circRNA_P1 / 2)
    err1 = np.sum(np.square(circRNA_p1-circRNA_P1_t))
    err2 = np.sum(np.square(circRNA_p2-circRNA_P2_t))
    err3 = np.sum(np.square(circRNA_p3 - circRNA_P3_t))

    if(i == m):
        break
    # if(err1 < 1e-6) and (err2 < 1e-6) and (err2 < 1e-6):
    #     print("circRNA迭代次数：",i)
    #     break
    circRNA_P2_t = circRNA_p2
    circRNA_P1_t = circRNA_p1
    circRNA_P3_t = circRNA_p3

circRNA_sl = (1/3)*circRNA_p1+ (1/3)*circRNA_p2 + (1/3)*circRNA_p3
#circRNA_sl = (0.5)*circRNA_p3+ (0.5)*circRNA_p1
#circRNA_sl = circRNA_p3
print(circRNA_sl, circRNA_sl.shape)
#print(circRNA_sl1)

disease_P2_t=disease_P2
disease_P1_t=disease_P1
disease_P3_t = disease_P3
for j in range(1000):
    disease_p1=alpha_1*(disease_S1@(disease_P2_t/2)@disease_S1.T)+(1-alpha_1)*(disease_P2/2)
    disease_p2=alpha_1*(disease_S2@(disease_P1_t/2)@disease_S2.T)+(1-alpha_1)*(disease_P1/2)
    disease_p3 =alpha_1*(disease_S3@(disease_P1_t/2)@disease_S3.T)+(1-alpha_1)*(disease_P1/2)
    err1 = np.sum(np.square(disease_p1-disease_P1_t))
    err2= np.sum(np.square(disease_p2-disease_P2_t))
    err3= np.sum(np.square(disease_p3-disease_P3_t))
    if(j==m):
        break
    # if (err1 < 1e-6) and (err2 < 1e-6) and (err3 < 1e-6):
    #     print("disease迭代的次数：", i)
    #     break
    disease_P2_t=disease_p2
    disease_P1_t=disease_p1
    disease_P3_t = disease_p3


#disease_sl = disease_p3
disease_sl = (1/3)*disease_p1 + (1/3)*disease_p2 + (1/3)*disease_p3
#disease_sl=0.5*disease_p3+0.5*disease_p1
print(disease_sl,disease_sl.shape)
#print(disease_sl1,disease_sl1.shape)

def compute_weightd_matrix(S1, k1):
    #compute N_j and N_i
    N_i=calculate_neighbors(S1, k1)
    N_j=calculate_neighbors(S1.T,k1)
    #w matirx
    w = np.zeros((len(S1), len(S1)))

    for i in range(len(S1)):
        for j in range(len(S1)):
            if i in N_j[j] and j in N_i[i]:
                w[i][j]=1
            elif i not in N_j[j] and j not in N_i[i]:
                w[i][j]=0
            else:
                w[i][j]=0.5

    return w

w1 = compute_weightd_matrix(circRNA_sl, K1)
w2 = compute_weightd_matrix(disease_sl, K2)

#average_circRNA= w1@circRNA_sl
#average_disease= w2@disease_sl
average_circRNA = circRNA_sl
average_disease = disease_sl

print(average_disease,average_disease.shape)
print(average_circRNA,average_circRNA.shape)

np.savetxt('circRNA_SMF-D1.txt', average_circRNA,fmt='%6f',delimiter='\t')
np.savetxt('disease_SMF-D1.txt', average_disease,fmt ='%6f', delimiter='\t')
#
#
