import random
import numpy as np

class ResiduaryResistance:
    def __init__(self, file_path='./data-set.txt', percent=0.8):
        file = open(file_path, 'r')
        lines = file.readlines()
        random.shuffle(lines)
        self.m = len(lines)
        self.n = len(lines[0].split())
        self.X = np.mat(np.zeros((self.m, self.n)))
        self.Y = np.mat(np.zeros((self.m, 1)))
        for i in range(self.m):
            line = lines[i].split()
            for j in range(self.n - 1):
                self.X[i, j] = float(line[j])
            self.X[i, self.n - 1] = 1.0
            self.Y[i, 0] = float(line[self.n - 1][:-1])
        self.m = int(self.m * percent)
        self.testX = self.X[self.m:,:]
        self.testY = self.Y[self.m:,:]
        self.X = self.X[:self.m,:]
        self.Y = self.Y[:self.m,:]
        
    def linear_regression(self):
        self.params = (self.X.T * self.X).I * self.X.T * self.Y

    def predict(self, data_in):
        return self.params.T * data_in.T

def test_run():
    rr = ResiduaryResistance()
    rr.linear_regression()
    for i in range(rr.testX.shape[0]):
        print('=====')
        print('guess: ', rr.predict(rr.testX[i])[0,0])
        print('actual: ', rr.testY[i,0])

test_run()
