import pandas as pd
import jieba
import pickle
import math
import os
import random

class ChineseTextClassification:
    def __init__(self):
        self.word_dict = {}
        # 初始化噪声词列表
        file = open('stopwords_cn.txt', 'r')
        self.stopword_list = file.readlines()
        file.close()
        self.stopword_list = [stopword.strip() for stopword in self.stopword_list]
        file = open('ClassList.txt', 'r')
        self.class_list = {}
        for line in file.readlines():
            line = line.split()
            self.class_list[line[1]] = line[0] # 在class_list中存储label: folder，folder为包含标记为label的所有文本的文件夹名
            self.word_dict.setdefault(line[1], {}) # 初始化标记为label的文本的词频字典
        file.close()

    def text_to_word_list(self, basic_path='./sample'):
        self.word_list = []
        self.label_list = []
        for label, folder in self.class_list.items():
            folder_path = os.path.join(basic_path, folder)
            filename_list = os.listdir(folder_path) # 标记为label的文件名的list
            for filename in filename_list:
                file_path = os.path.join(folder_path, filename)
                # label_list和word_list中相同下标的元素对应于同一个文件
                self.word_list.append(pickle.dumps(self.get_word_list(file_path))) # 将文件进行分词，并将分词得到的词语列表序列化为string加入word_list列表
                self.label_list.append(label) # 将文件的标记label加入label_list

    def get_word_list(self, file_path):
        file = open(file_path, 'r')
        raw_text = file.read()
        file.close()
        seg_list = jieba.cut(raw_text)
        word_list = []
        for word in seg_list:
            if word != '':
                word_list.append(word)
        dic = dict([(w, 1) for w in word_list]) # 将词汇作为dict中的key，便于删除噪声词
        dic = self.rm_stopword(dic)
        return dic

    def rm_stopword(self, dic):
        for word in self.stopword_list:
            if word in dic:
                del dic[word]
        return dic

    def train(self, percent=0.8):
        self.text_to_word_list()
        word_label_list = list(zip(self.word_list, self.label_list))
        random.shuffle(word_label_list)
        train_set = word_label_list[:int(len(word_label_list) * percent) + 1]
        test_set = word_label_list[int(len(word_label_list) * percent) + 1:]
        self.all_text_count = len(train_set) # 训练集中的文本总数
        self.label_text_count = {} # label: 训练集中标记为label的文本总数
        for word_list, label in train_set:
            self.label_text_count.setdefault(label, 0)
            self.label_text_count[label] += 1
            word_list = pickle.loads(word_list)
            for word in word_list:
                self.word_dict[label].setdefault(word, 0)
                self.word_dict[label][word] += 1
        print('Train successfully')
        return test_set

    def predict(self, word_list):
        p_label = {}
        for label, count in self.label_text_count.items():
            p_label[label] = math.log(float(count) / self.all_text_count)
        for word in word_list:
            for label in self.label_text_count:
                self.word_dict[label].setdefault(word, 0)
                p_word_label = self.word_dict[label][word]
                if p_word_label == 0:
                    p_label[label] += math.log(1.0 / (len(word_list) + 1) / self.all_text_count)
                else:
                    p_label[label] += math.log(float(p_word_label) / self.label_text_count[label])
        p_label = sorted(p_label.items(), key=lambda x: x[1], reverse=True)
        return p_label[0][0]

def test_run():
    ctc = ChineseTextClassification()
    test_set = ctc.train()
    print(len(test_set))
    wrong = 0
    right = 0
    for label, folder in ctc.class_list.items():
        folder_path = os.path.join('./sample', folder)
        filename_list = os.listdir(folder_path)
        for filename in filename_list:
            file_path = os.path.join(folder_path, filename)
            test_word_list = [i for i in ctc.get_word_list(file_path).keys()]
            if ctc.predict(test_word_list) != label:
                wrong += 1
            else:
                right += 1
    print(wrong)
    print(right)
    wrong = 0
    right = 0
    for case, label in test_set:
        if ctc.predict(pickle.loads(case)) != label:
            wrong += 1
        else:                
            right += 1
    print(wrong)
    print(right)

test_run()
