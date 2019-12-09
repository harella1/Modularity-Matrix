import numpy as np
from numpy import linalg as LA


def GetEigen_WithWeights (Prov_mat, Cons_mat, Prov_Weight, Cons_Weight):
    dim = Prov_mat.size
    union_mat = Prov_mat*Prov_Weight + Cons_mat*Cons_Weight
    row_sum = np.sum(union_mat,axis = 1, dtype=int)
    col_sum = np.sum(union_mat,axis = 0, dtype=int)
    deg_mat = np.diag(np.concatenate((row_sum,col_sum)))
    adj_mat = np.block([[np.zeros((dim,dim)), union_mat], [union_mat.T,np.zeros((dim,dim))]])
    adj_mat = np.block([[np.zeros((dim,dim)),union_mat.T], [union_mat, np.zeros((dim,dim))]])
    laplacian = np.matrix(deg_mat-adj_mat, dtype=int)
    v, w = LA.eig(laplacian)
    return v,w