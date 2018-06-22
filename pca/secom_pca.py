import numpy as np
import math

def npnan(x):
    if x == float('NaN'): 
        return np.nan

class SecomPCA:
    def __init__(self, file_path='secom.data'):
        file = open(file_path, 'r')
        temp_array = [line.strip().split() for line in file.readlines()]
        for line in temp_array:
            for i in line:
                i = float(i)
                if i == float('NaN'):
                    i = np.nan
        self.X = np.mat(temp_array, dtype=float)
        for i in range(self.X.shape[1]):
            self.X[np.nonzero(np.isnan(self.X[:,i].A))[0], i] = np.mean(self.X[np.nonzero(~np.isnan(self.X[:,i].A))[0], i])
    
    def pca(self, n):
        mean_mat = np.mean(self.X, axis=0)
        self.X = self.X - mean_mat
        eig_vals, eig_vecs = np.linalg.eig(np.mat(np.cov(self.X, rowvar=False)))
        eig_vals_idx = np.argsort(eig_vals)[:-(n + 1):-1]
        eig_vecs = eig_vecs[:,eig_vals_idx]
        low_d_X = self.X * eig_vecs
        return low_d_X

def test_run():
    sp = SecomPCA()
    print(sp.pca(3))

test_run()
