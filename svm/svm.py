import numpy as np
import random

linear_kernel = lambda Xi, Xj: Xi * Xj.T

class SVM:
    def __init__(self, file_path='./testSet.txt', C=1.0, toler=1e-3, percent=0.8):
        file = open(file_path, 'r')
        lines = file.readlines()
        random.shuffle(lines)
        self.m = len(lines)
        self.n = len(lines[0].split('\t')) - 1
        self.X = np.mat(np.zeros((self.m, self.n)))
        self.Y = np.mat(np.zeros((self.m, 1)))
        for i in range(self.m):
            line = lines[i].split('\t')
            if line[-1] == '-1\n':
                self.Y[i][0] = -1.0
            elif line[-1] == '1\n':
                self.Y[i][0] = 1.0
            for j in range(self.n):
                self.X[i, j] = float(line[j])
        self.b = 0
        self.C = C
        self.toler = toler
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.e_cache = np.mat(np.zeros((self.m, 2)))
        self.testX = self.X[int(self.m * percent):,:]
        self.testY = self.Y[int(self.m * percent):,:]

    def smo(self, K=linear_kernel, iter_times=10):
        iter = 0
        is_changed = True
        is_entire_loop = True
        while iter < iter_times and (is_changed or is_entire_loop):
            print(iter)
            is_changed = False
            if is_entire_loop:
                idx_list = list(range(self.m))
            else:
                idx_list = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
            for i in idx_list:
                ei = self.calc_ek(i)
                if ((self.Y[i] * ei < -self.toler) and (self.alphas[i] < self.C)) or ((self.Y[i] * ei > self.toler) and (self.alphas[i] > 0)):
                    j, ej = self.select_j(i, ei)
                    old_ai = self.alphas[i].copy()
                    old_aj = self.alphas[j].copy()
                    if self.Y[i] != self.Y[j]:
                        L = max(0, old_aj - old_ai)
                        H = min(self.C, self.C + old_aj - old_ai)
                    else:
                        L = max(0, old_aj + old_ai - self.C)
                        H = min(self.C, old_aj + old_ai)
                    if L == H:
                        continue
                    eta = K(self.X[i,:], self.X[i,:]) + K(self.X[j,:], self.X[j,:]) - 2 * K(self.X[i,:], self.X[j,:])
                    if eta <= 0:
                        continue
                    self.alphas[j, 0] += self.Y[j, 0] * (ei - ej) / eta
                    self.alphas[j] = self.clip_alpha(self.alphas[j], L, H)
                    self.update_ek(j)
                    if abs(self.alphas[j] - old_aj) < self.toler:
                        continue
                    self.alphas[i] += self.Y[j] * self.Y[i] * (old_aj - self.alphas[j])
                    self.update_ek(i)
                    b1 = self.b - ei - self.Y[i] * (self.alphas[i] - old_ai) * K(self.X[i], self.X[i]) - self.Y[j] * (self.alphas[j] - old_aj) * K(self.X[i], self.X[j])
                    b2 = self.b - ej - self.Y[i] * (self.alphas[i] - old_ai) * K(self.X[i], self.X[j]) - self.Y[j] * (self.alphas[j] - old_aj) * K(self.X[j], self.X[j])
                    if 0 < self.alphas[i] and self.C > self.alphas[i]:
                        self.b = b1
                    elif 0 < self.alphas[j] and self.C > self.alphas[j]:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2 
                    is_changed = True
            if is_entire_loop:
                is_entire_loop = False
            elif is_changed == False:
                is_entire_loop = True
            iter += 1

    def predict(self, test_in):
        return np.multiply(self.alphas, self.Y).T * (self.X * test_in.T) + self.b                  
        
    def calc_ek(self, k):
        fxk = np.multiply(self.alphas, self.Y).T * (self.X * self.X[k,:].T) + self.b
        return fxk - self.Y[k]
    
    def update_ek(self, k):
        ek = self.calc_ek(k)
        self.e_cache[k] = [1, ek]

    def select_j(self, i, ei):
        max_delta_e = -1
        j = -1
        ej = -1
        valid_ecache_list = np.nonzero(self.e_cache[:,0].A)[0]
        if len(valid_ecache_list) > 1:
            for k in valid_ecache_list:
                if k == i:
                    continue
                delta_e = abs(ei - self.calc_ek(k))
                if delta_e > max_delta_e:
                    max_delta_e = delta_e
                    j = k
                    ej = self.e_cache[k, 1]
            return j, ej
        else:
            j = i
            while j == i:
                j = int(random.uniform(0, self.m))
            ej = self.calc_ek(j)
            return j, ej
    
    def clip_alpha(self, aj, L, H):
        if H < aj:
            aj = H
        if L > aj:
            aj = L
        return aj

def test_run(file_path='./test-set/test.txt'):
    svm = SVM()
    svm.smo()
    right = 0
    wrong = 0
    for i in range(svm.testX.shape[0]):
        if svm.testY[i][0] * svm.predict(svm.testX[i]) > 0:
            right += 1
        else:
            wrong += 1
    print(float(right) / (right + wrong))

test_run()
