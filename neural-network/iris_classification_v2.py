import random
import numpy as np

def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))

class IrisClassification:
    def __init__(self, file_path='./iris.data', cell_num=[4, 10, 3], percent=0.8):
        self.cell_num = cell_num[:] # cell_num的默认参数表示输入层有4个节点，隐层有10个节点，输出层有3个节点
        file = open(file_path, 'r')
        lines = file.readlines()
        file.close()
        random.shuffle(lines)
        self.m = len(lines)
        self.n = cell_num[0]
        self.X = np.mat(np.zeros((self.m, self.n)))
        self.Y = np.mat(np.zeros((self.m, cell_num[2])))
        for i in range(self.m):
            line = lines[i].split(',')
            for j in range(self.n):
                self.X[i, j] = float(line[j])
            if line[-1][:-1] == 'Iris-setosa':
                self.Y[i, 0] = 1.0
            elif line[-1][:-1] == 'Iris-versicolor':
                self.Y[i, 1] = 1.0
            elif line[-1][:-1] == 'Iris-virginica':
                self.Y[i, 2] = 1.0
        self.w = np.mat(np.random.random((cell_num[2], cell_num[1])))
        self.theta = np.mat(np.random.random((cell_num[2], 1)))
        self.v = np.mat(np.random.random((cell_num[1], cell_num[0])))
        self.gama = np.mat(np.random.random((cell_num[1], 1)))
        self.m = int(self.m * percent)
        self.testX = self.X[self.m:,:]
        self.testY = self.Y[self.m:,:]
        self.X = self.X[:self.m,:]
        self.Y = self.Y[:self.m,:]

    def nn(self, iter_times=500, yita=0.1):
        for it in range(iter_times):
            for i in range(self.m):
                alpha = self.v * self.X[i].T
                b = alpha - self.gama
                b = sigmod(b)
                beta = self.w * b
                y = beta - self.theta
                y = sigmod(y)
                g = np.multiply(np.multiply(y, (1 - y)), (self.Y[i].T - y))
                e = np.multiply(np.multiply(b, (1 - b)), self.w.T * g)
                self.w = self.w + yita * g * b.T
                self.theta = self.theta - yita * g
                self.v = self.v + yita * e * self.X[i]
                self.gama = self.gama - yita * e

    def predict(self, data):
        alpha = self.v * data
        b = alpha - self.gama
        b = sigmod(b)
        beta = self.w * b
        y = beta - self.theta
        y = sigmod(y)
        if y[0, 0] > y[1, 0] and y[0, 0] > y[2, 0]:
            return 'Iris-setosa'
        elif y[1, 0] > y[0, 0] and y[1, 0] > y[2, 0]:
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'

def test_run():
    ic = IrisClassification()
    ic.nn()
    right = 0
    wrong = 0
    for i in range(ic.testX.shape[0]):
        label = ic.predict(ic.testX[i].T)
        if (label == 'Iris-setosa' and ic.testY[i,0] > 0.5) or (label == 'Iris-versicolor' and ic.testY[i,1] > 0.5) or (label == 'Iris-virginica' and ic.testY[i,2] > 0.5):
            right += 1
        else:
            wrong += 1
    print(float(right) / (right + wrong))

test_run()
    