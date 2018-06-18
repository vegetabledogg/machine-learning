import math
import random

class IrisClassification:
    def __init__(self, file_path='./iris.data', percent=0.8):
        self.data_set = []
        file = open(file_path, 'r')
        for line in file.readlines():
            line = line.split(',')
            for i in range(4):
                line[i] = float(line[i])
            self.data_set.append(line)
        file.close()
        random.shuffle(self.data_set)
        self.test_set = self.data_set[int(len(self.data_set) * percent) + 1:]
        self.data_set = self.data_set[:int(len(self.data_set) * percent) + 1]
        self.attr_set = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
        self.root = self.create_node(self.data_set[:])

    def calc_ent(self, data_set):
        total_data = len(data_set)
        label_dict = {}
        for data in data_set:
            label_dict.setdefault(data[-1], 0)
            label_dict[data[-1]] += 1
        ent = 0
        for v in label_dict.values():
            pk = float(v) / total_data
            ent -= pk * math.log(pk)
        return ent

    def split_data_set(self, data_set, index, t):
        data_set_1 = []
        data_set_2 = []
        for data in data_set:
            temp_data = data[:index]
            temp_data.extend(data[index + 1:])
            if data[index] < t:
                data_set_1.append(temp_data)
            else:
                data_set_2.append(temp_data)
        return data_set_1, data_set_2

    def choose_t(self, data_set):
        max_gain = 0
        max_index = -1
        max_t = -1
        ent = self.calc_ent(data_set)
        total_data = len(data_set)
        for i in range(len(data_set[0]) - 1):
            attr_list = [data[i] for data in data_set]
            attr_list.sort()
            for j in range(len(attr_list) - 1):
                t = (float(attr_list[j]) + float(attr_list[j + 1])) / 2.0
                data_set_1, data_set_2 = self.split_data_set(data_set, i, t)
                ent_1 = self.calc_ent(data_set_1)
                ent_2 = self.calc_ent(data_set_2)
                gain = ent - float(len(data_set_1) / total_data) * ent_1 - float(len(data_set_2) / total_data) * ent_2
                if max_gain < gain:
                    max_gain = gain
                    max_index = i
                    max_t = t
        return max_index, max_t

    def create_node(self, data_set):
        label_list = [data[-1] for data in data_set]
        if label_list.count(label_list[0]) == len(label_list):
            return label_list[0]
        if len(data_set[0]) == 1:
            return self.get_max_label(label_list)
        index, t = self.choose_t(data_set)
        data_set_1, data_set_2 = self.split_data_set(data_set, index, t)
        return {'index': index, 't': t, 'left': self.create_node(data_set_1), 'right': self.create_node(data_set_2)}

    def get_max_label(self, label_list):
        label_count = {}
        for label in label_list:
            label_count.setdefault(label, 0)
            label_count[label] += 1
        label_count = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
        return label_count[0][0]

    def predict(self, data):
        node = self.root
        while True:
            if data[node['index']] < node['t']:
                if isinstance(node['left'], dict):
                    del(data[node['index']])
                    node = node['left']
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    del(data[node['index']])
                    node = node['right']
                else:
                    return node['right']
            

def test_run():
    ic = IrisClassification()
    right = 0
    wrong = 0
    for data in ic.test_set:
        if data[-1] == ic.predict(data):
            right += 1
        else:
            wrong += 1
    print(float(right) / (right + wrong))

test_run()
