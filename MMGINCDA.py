import numpy as np

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_curve
import copy
import Global_similarity
import Local_similarity
import DGI
import plot_roc
import random

K1 = 5
K2 = 5

CC=np.loadtxt(r"circRNA_SMF-D1.txt", dtype = float)
DD = np.loadtxt(r"disease_SMF-D1.txt", dtype = float)
Y = np.loadtxt(r"association1.txt", dtype=float)
print(Y.shape, np.sum(Y==1))
circRNA_disease_k = np.loadtxt(r"known-D1.txt", dtype = int)
circRNA_disease_uk = np.loadtxt(r"unknown-D1.txt",dtype =int)

def DC(D, mu, T0, g):
    U,S,V=np.linalg.svd(D)
    T1 = np.zeros(np.size(T0))
    for i in range(1,100):
        T1 = DCInner(S, mu, T0, g)
        err = np.sum(np.square(T1-T0))
        if err < 1e-6:
            break
        T0= T1

    V = V[:585,:]
    l_1 = np.dot(U, np.diag(T1))
    #print(l_1.shape,V.shape)
    l=np.dot(l_1, V)
    return l,T1

def DCInner(S, mu, T_k, gam):
    lamb = 1/mu
    grad = (1+gam)*gam/(np.square(gam+T_k))
    T_k1 = S-lamb*grad
    T_k1[T_k1<0] =0
    return T_k1

def GAMA(H, A,B):
    muzero = 15
    mu = muzero
    gamma = 0.06
    rho = 2
    tol = 1e-3
    alpha = 2

    m,n = np.shape(H)
    L = copy.deepcopy(H)
    S = np.zeros((m,n))
    Y = np.zeros((m,n))

    omega = np.zeros(H.shape)
    omega[H.nonzero()] = 1

    for i in range(0,500):
        tran = (1/mu)*(Y+alpha*(H*omega) + np.dot(A,B)) + L
        w = tran - (alpha/(alpha+mu))*omega*tran
        w[w<0]=0
        w[w>1]=1

        D=w-Y/mu
        sig = np.zeros(min(m,n))
        L, sig = DC(copy.deepcopy(D), mu,copy.deepcopy(sig), gamma)

        #Y
        Y = Y + mu*(L-w)
        mu = mu*rho
        sigma = np.linalg.norm(L-w,'fro')
        PRE = sigma/np.linalg.norm(H, 'fro')
        if PRE < tol:
            break
    return w
#120
def truncated(H0):
    for i in range(0,2):
        U,S,V = np.linalg.svd(H0)
        #print(U.shape,V.shape)
        r = 120
        A = U[:,:r]
        B = V[:r,:]

        H0 = GAMA(H0,A,B)

    Smmi = H0
    return Smmi


#def plot_Roc():

def main():
    roc_sum,acc_sum,f1_sum,pre_sum,rec_sum,aupr_sum, time = 0,0,0,0,0,0,0
    kf = KFold(n_splits=5, shuffle=True, random_state=99999)
    for train_index, test_index in kf.split(circRNA_disease_k):
        x_2 = copy.deepcopy(Y)
        #print('train_index:',train_index)
        for index in test_index:
            x_2[circRNA_disease_k[index,0], circRNA_disease_k[index,1]] = 0

        #global similarity
        G_circRNA = Global_similarity.compute_GS_matrix(CC)
        G_disease = Global_similarity.compute_GS_matrix(DD)
        #G_circRNA = CC
        #G_disease = DD
        Yh1 = DGI.fHGI(0.1, G_circRNA, G_disease, x_2)

        L_circRNA = Local_similarity.row_normalization(CC, K1)
        L_disease = Local_similarity.row_normalization(DD, K1)
        #L_circRNA=CC
        #L_disease=DD
        Yh2 = DGI.fHGI(0.3, L_circRNA,L_disease, x_2)

        L_1 = (Yh1 + Yh2)/2
        #print('L1:',L_1.shape)
        H = np.hstack((CC, L_1))
       # print('H:',H.shape)
        M_1 = truncated(H)

        M_1 = M_1[0:CC.shape[0], CC.shape[0]:H.shape[1]]
        #Label = np.zeros(circRNA_disease_uk.shape[0]+test_index.size)
        #Score = np.zeros(circRNA_disease_uk.shape[0]+test_index.size)
        unknow_index_size =test_index.size
        Label = np.zeros(unknow_index_size + test_index.size)
        Score = np.zeros(unknow_index_size + test_index.size)
        np.savetxt('circRNA_disease-D1-predicted.txt', M_1-Y, fmt='%6f', delimiter='\t')
        i,j = 0,0
        for s_index in test_index:
            Label[i]=1
            Score[i]=M_1[circRNA_disease_k[s_index,0],circRNA_disease_k[s_index,1]]
            i= i+1

        #for i in range(test_index.size, circRNA_disease_uk.shape[0] + test_index.size):
        u_m = len(circRNA_disease_uk)
        u_m1 = random.sample(range(u_m), unknow_index_size)

        for i in range(test_index.size, unknow_index_size + test_index.size):
            j=u_m1[i-test_index.size]
            Score[i] = M_1[circRNA_disease_uk[j, 0], circRNA_disease_uk[j, 1]]



        fpr, tpr, thersholds = roc_curve(y_true=Label, y_score=Score, drop_intermediate=False)
        TP = np.sum((Label==1)&(Score>0.01))
        #print(Label, Score)
        # 计算TP, FP, FN
        TP = np.sum((Label == 1) & (Score > 0.01))
        TN = np.sum((Label == 0) & (Score < 0.01))
        FP = np.sum((Label == 0) & (Score > 0.01))
        FN = np.sum((Label == 1) & (Score < 0.01))
        ACC = (TP+TN)/(TP+TN+FP+FN)
        pre = TP/(TP+FP)
        rec = TP/(TP+FN)
        f1_1 = 2 * (pre * rec) / (pre + rec)
        #print(ACC,pre,rec,f1_1)
        roc_auc = auc(fpr, tpr)
        roc_sum = roc_sum + roc_auc

        #print(Label, Score)
        precision, recall, thresholds = precision_recall_curve(Label, Score)
        aupr = auc(recall, precision)
        #f1 = 2.0 * (precision * recall) / (precision + recall)

        #plot_roc.plot_ROC(fpr, tpr, roc_auc)
        acc_sum = acc_sum + ACC
        f1_sum = f1_sum+ f1_1
        pre_sum = pre_sum+pre
        rec_sum = rec_sum+rec
        aupr_sum = aupr_sum+aupr

        time += 1
        mean1 = np.mean(
            [roc_sum / time, aupr_sum / time, acc_sum / time, rec_sum / time, pre_sum / time, f1_sum / time])
        s = (roc_sum / time, aupr_sum / time, acc_sum / time, rec_sum / time, pre_sum / time, f1_sum / time, mean1)
        print(time, ACC, pre, rec, f1_1, aupr, roc_auc, roc_sum, s)
        while (time == 5):
            return s#

if __name__ == "__main__":
    total,time= 0,0

    for i in range(0,100):
        l=main()
        print("\n")
        print(l)
        break
