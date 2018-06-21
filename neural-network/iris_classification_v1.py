import random
import numpy as np

def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))

class IrisClassification:
    def __init__(self, file_path='./iris.data', cell_num=[4, 10, 3], percent=0.8):
        self.cell_num = cell_num[:] # cell_num的默认参数表示输入层有4个节点，隐层有10个节点，输出层有3个节点
        self.data_set = []
        file = open(file_path, 'r')
        for line in file.readlines():
            line = line.split(',')
            for i in range(len(line) - 1):
                line[i] = float(line[i])
            if line[-1] == 'Iris-setosa\n':
                line[-1] = [1, 0, 0]
            elif line[-1] == 'Iris-versicolor\n':
                line[-1] = [0, 1, 0]
            elif line[-1] == 'Iris-virginica\n':
                line[-1] = [0, 0, 1]
            self.data_set.append(line)
        file.close()
        random.shuffle(self.data_set)
        self.test_set = self.data_set[int(len(self.data_set) * 0.8):]
        self.data_set = self.data_set[:int(len(self.data_set) * 0.8)]
        self.weight_list = [] # weight_list[i]表示第i层和第i+1层节点之间的权值列表，权值列表中权值的排列顺序为w11, w12, ..., w1n, w21, ...
        for i in range(len(cell_num) - 1):
            temp_weight_list = []
            for j in range(cell_num[i] * cell_num[i + 1]):
                temp_weight_list.append(random.random())
            self.weight_list.append(temp_weight_list)
        self.threshold_list = [] # threshold_list[i]表示第i+1层节点的阈值列表
        for i in range(len(cell_num) - 1):
            temp_threshold_list = []
            for j in range(cell_num[i + 1]):
                temp_threshold_list.append(random.random())
            self.threshold_list.append(temp_threshold_list)

    def nn(self, iter_times=500, yita=0.1):
        for it in range(iter_times):
            for data in self.data_set:
                output_dict = dict((idx, data[idx]) for idx in range(self.cell_num[0]))
                y = data[-1]
                for j in range(1, len(self.cell_num)):
                    input_dict = {}
                    for out_idx in range(self.cell_num[j - 1]):
                        for in_idx in range(self.cell_num[j]):
                            weight = self.weight_list[j - 1][out_idx * self.cell_num[j] + in_idx]
                            input_dict.setdefault(in_idx, 0.0)
                            input_dict[in_idx] += weight * output_dict[out_idx]
                    last_output_dict = output_dict
                    output_dict = {}
                    for in_idx in range(self.cell_num[j]):
                        threshold = self.threshold_list[j - 1][in_idx]
                        output_dict[in_idx] = sigmod(input_dict[in_idx] - threshold)
                ek = self.calc_ek(output_dict.values(), y)
                g_list = []
                for j in range(self.cell_num[2]):
                    g_list.append(output_dict[j] * (1 - output_dict[j]) * (y[j] - output_dict[j])) 
                for h in range(self.cell_num[1]):
                    temp_sum = 0.0
                    for j in range(self.cell_num[2]):
                        temp_sum += self.weight_list[1][h * self.cell_num[2] + j] * g_list[j]
                    eh = last_output_dict[h] * (1 - last_output_dict[h]) * temp_sum
                    for i in range(self.cell_num[0]):
                        self.weight_list[0][i * self.cell_num[1] + h] += yita * eh * data[i]
                        self.threshold_list[0][h] -= yita * eh
                for j in range(self.cell_num[2]):
                    for h in range(self.cell_num[1]):
                        self.weight_list[1][h * self.cell_num[2] + j] += yita * g_list[j] * last_output_dict[h]
                        self.threshold_list[1][j] -= yita * g_list[j]

    def calc_ek(self, vec_1, vec_2):
        ek = 0.0
        for i, j in zip(vec_1, vec_2):
            ek += (i - j) ** 2
        return ek / 2

    def predict(self, data):
        output_dict = dict((idx, data[idx]) for idx in range(self.cell_num[0]))
        for j in range(1, len(self.cell_num)):
            input_dict = {}
            for out_idx in range(self.cell_num[j - 1]):
                for in_idx in range(self.cell_num[j]):
                    weight = self.weight_list[j - 1][out_idx * self.cell_num[j] + in_idx]
                    input_dict.setdefault(in_idx, 0.0)
                    input_dict[in_idx] += weight * output_dict[out_idx]
            output_dict = {}
            for in_idx in range(self.cell_num[j]):
                threshold = self.threshold_list[j - 1][in_idx]
                output_dict[in_idx] = sigmod(input_dict[in_idx] - threshold)
        if output_dict[0] > output_dict[1] and output_dict[0] > output_dict[2]:
            return 'Iris-setosa'
        elif output_dict[1] > output_dict[0] and output_dict[1] > output_dict[2]:
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'

def test_run():
    ic = IrisClassification()
    ic.nn()
    right = 0
    wrong = 0
    for data in ic.test_set:
        label = ic.predict(data)
        if (label == 'Iris-setosa' and data[-1] == [1, 0, 0]) or (label == 'Iris-versicolor' and data[-1] == [0, 1, 0]) or (label == 'Iris-virginica' and data[-1] == [0, 0, 1]):
            right += 1
        else:
            wrong += 1
    print(float(right) / (right + wrong))

test_run()
    