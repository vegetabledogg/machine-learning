import numpy as np
import math
import random

class DiabetesDiagnose:
    def __init__(self, file_path='./data-set.txt', percent=0.8):
        file = open(file_path, 'r')
        lines = file.readlines()
        random.shuffle(lines)
        self.m = len(lines)
        self.n = len(lines[0].split(','))
        self.X = np.mat(np.zeros((self.m, self.n)))
        self.Y = np.mat(np.zeros((self.m, 1)))
        for i in range(self.m):
            line = lines[i].split(',')
            for j in range(self.n - 1):
                self.X[i, j] = float(line[j])
            self.X[i, self.n - 1] = 1.0
            self.Y[i, 0] = float(line[self.n - 1][:-1])
        self.m = int(self.m * percent)
        self.testX = self.X[self.m:,:]
        self.testY = self.Y[self.m:,:]
        self.X = self.X[:self.m,:]
        self.Y = self.Y[:self.m,:]
        self.betas = np.mat(np.zeros((self.n, 1)))

    def logistic_regression(self, toler=1e-2):
        likelyhood = self.get_likelyhood()
        while True:
            d_1 = np.mat(np.random.random((self.n, 1)))
            d_2 = 0.0
            for i in range(self.m):
                f = self.betas.T * self.X[i].T
                p_1 = math.exp(f) / (1 + math.exp(f))
                d_1 = d_1 - self.X[i].T * (self.Y[i, 0] - p_1)
                d_2 = d_2 + self.X[i] * self.X[i].T * p_1 * (1 - p_1)
            self.betas = self.betas - d_1 / d_2
            new_likelyhood = self.get_likelyhood()
            if abs(new_likelyhood - likelyhood) < toler:
                break
            else:
                likelyhood = new_likelyhood

    def get_likelyhood(self):
        likelyhood = 0.0
        for i in range(self.m):
            f = self.betas.T * self.X[i].T
            likelyhood += (-self.Y[i, 0] * f + math.log(1 + math.exp(f)))
        return likelyhood

    def predict(self, data_in):
        return 1 / (1 + math.exp(-(self.betas.T * data_in)))

def test_run():
    dd = DiabetesDiagnose()
    dd.logistic_regression()
    right = 0
    wrong = 0
    for i in range(dd.testX.shape[0]):
        res = dd.predict(dd.testX[i].T)
        if (dd.testY[i, 0] > 0.5 and res > 0.5) or (dd.testY[i, 0] < 0.5 and res < 0.5):
            right += 1
        else:
            wrong += 1
    print(float(right) / (wrong + right))

np.seterr(divide='ignore',invalid='ignore')
test_run()
