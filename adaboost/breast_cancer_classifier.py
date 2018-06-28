import numpy as np
import math
import random

class BreastCancerClassifier:
    def __init__(self, file_path='./data-set.txt', percent=0.8):
        file = open(file_path, 'r')
        lines = file.readlines()
        file.close()
        random.shuffle(lines)
        self.m = len(lines)
        self.n = len(lines[0].split(','))
        self.X = np.mat(np.zeros((self.m, self.n)))
        for i in range(self.m):
            line = lines[i][:-1].split(',')
            for j in range(self.n):
                self.X[i, j] = float(line[j])
        self.testX = self.X[int(self.m * percent):]
        self.X = self.X[:int(self.m * percent)]
        self.testm = self.m - int(self.m * percent)
        self.m = int(self.m * percent)
        self.D = np.mat(np.ones((self.m, 1)) / self.m) 

    def decision_tree(self):
        min_err = None
        min_weighted_err = 1000000
        min_op = 0
        min_sep = 0
        min_idx = 0
        for i in range(self.n - 1):
            attr_list = self.X[:,i]
            attr_list.sort(axis=0)
            for j in range(self.m - 1):
                err = np.mat(np.ones((self.m, 1)))
                sep = (attr_list[j, 0] + attr_list[j + 1, 0]) / 2
                for op in ['gt', 'lt']:
                    for k in range(self.m):
                        if op == 'gt':
                            if (self.X[k, i] > sep and self.X[k, self.n - 1] > 1.5) or (self.X[k, i] <= sep and self.X[k, self.n - 1] < 1.5):
                                err[k, 0] = 0
                        else:
                            if (self.X[k, i] < sep and self.X[k, self.n - 1] > 1.5) or (self.X[k, i] >= sep and self.X[k, self.n - 1] < 1.5):
                                err[k, 0] = 0
                    weighted_err = err.T * self.D
                    if weighted_err[0,0] < min_weighted_err:
                        min_weighted_err = weighted_err
                        min_op = op
                        min_sep = sep
                        min_idx = i
                        min_err = err.copy()
        return min_weighted_err, min_err, min_op, min_sep, min_idx

    def adaboost(self, T):
        rate = 0
        self.weak_classifier = []
        self.alphas = np.mat(np.zeros((T, 1)))
        for i in range(T):
            classifier = {}
            w_err, err, op, sep, idx = self.decision_tree()
            if w_err > 0.5:
                break
            self.alphas[i, 0] = 0.5 * math.log((1 - w_err) / max(w_err, 1e-16))
            temp_mat = np.mat(np.ones((self.m, 1)))
            temp_mat[err[:,0] < 0.5] = -1.0
            self.D = np.multiply(self.D, np.exp(self.alphas[i, 0] * temp_mat))
            self.D = self.D / self.D.sum()
            classifier['op'] = op
            classifier['sep'] = sep
            classifier['idx'] = idx
            self.weak_classifier.append(classifier)
            temp_rate = self.test(i + 1)
            if temp_rate > rate:
                rate = temp_rate
            else:
                break

    def predict(self, data_in):
        res = 0
        for i in range(len(self.weak_classifier)):
            classifier = self.weak_classifier[i]
            if classifier['op'] == 'gt':
                if data_in[classifier['idx'], 0] > classifier['sep']:
                    res += self.alphas[i] * 1.0
                else:
                    res += self.alphas[i] * -1.0
            else:
                if data_in[classifier['idx'], 0] < classifier['sep']:
                    res += self.alphas[i] * 1.0
                else:
                    res += self.alphas[i] * -1.0
        if res > 0:
            return 2.0
        else:
            return 1.0

    def test(self, T):
        right = 0.0
        wrong = 0.0
        for i in range(self.testm):
            if (self.testX[i, self.n - 1] > 1.5 and self.predict(self.testX[i].T) > 1.5) or (self.testX[i, self.n - 1] < 1.5 and self.predict(self.testX[i].T) < 1.5):
                right += 1
            else:
                wrong += 1
        return right / (wrong + right)

def test_run(T=100):
    bcc = BreastCancerClassifier()
    bcc.adaboost(T)
    right = 0.0
    wrong = 0.0
    for i in range(bcc.testm):
        if (bcc.testX[i, bcc.n - 1] > 1.5 and bcc.predict(bcc.testX[i].T) > 1.5) or (bcc.testX[i, bcc.n - 1] < 1.5 and bcc.predict(bcc.testX[i].T) < 1.5):
            right += 1
        else:
            wrong += 1
    print(right / (wrong + right))

test_run()
                        