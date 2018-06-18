import pandas as pd
import jieba
import pickle
import math
import os
import random

class ChineseTextClassification:
    def __init__(self, percent=0.8, initial=1):
        # 初始化噪声词列表
        file = open('stopwords_cn.txt', 'r')
        self.stopword_list = file.readlines()
        file.close()
        self.stopword_list = [stopword.strip() for stopword in self.stopword_list]
        if initial == 0: # 分类器未进行初始化
            self.word_dict = {} 
            file = open('ClassList.txt', 'r')
            self.class_list = {}
            for line in file.readlines():
                line = line.split()
                self.class_list[line[1]] = line[0] # 在class_list中存储label: folder，folder为包含标记为label的所有文本的文件夹名
                self.word_dict.setdefault(line[1], {}) # 初始化标记为label的文本的词频字典
            self.train(percent)
        else:
            file = open('word_dict', 'rb')
            self.word_dict = pickle.load(file)
            file.close()
            file = open('all_text_count', 'rb')
            self.all_text_count = pickle.load(file)
            file.close()
            file = open('label_text_count', 'rb')
            self.label_text_count = pickle.load(file)
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

    def train(self, percent):
        self.text_to_word_list()
        word_label_list = list(zip(self.word_list, self.label_list))
        random.shuffle(word_label_list)
        self.train_set = word_label_list[:int(len(word_label_list) * percent) + 1]
        self.test_set = word_label_list[int(len(word_label_list) * percent) + 1:]
        self.all_text_count = len(self.train_set) # 训练集中的文本总数
        self.label_text_count = {} # label: 训练集中标记为label的文本总数
        for word_list, label in self.train_set:
            self.label_text_count.setdefault(label, 0)
            self.label_text_count[label] += 1 # 训练集中标记为label的文本总数增加1
            word_list = pickle.loads(word_list)
            for word in word_list:
                self.word_dict[label].setdefault(word, 0)
                self.word_dict[label][word] += 1 # 标记为label的文本中word的词频增加1
        file = open('word_dict', 'wb')
        self.word_dict = pickle.dump(self.word_dict, file, 2)
        file.close()
        file = open('all_text_count', 'wb')
        self.all_text_count = pickle.dump(self.all_text_count, file, 2)
        file.close()
        file = open('label_text_count', 'wb')
        self.label_text_count = pickle.dump(self.label_text_count, file, 2)
        file.close()
        print('Train successfully')

    def predict(self, word_list):
        p_label = {}
        for label, count in self.label_text_count.items():
            p_label[label] = math.log(float(count) / self.all_text_count) # log(p(label))
            for word in word_list:
                self.word_dict[label].setdefault(word, 0)
                p_word_label = self.word_dict[label][word]
                if p_word_label == 0:
                    p_label[label] += math.log(1.0 / (len(word_list) + 1) / self.all_text_count) # log(p(xi|label))
                else:
                    p_label[label] += math.log(float(p_word_label) / count) # log(p(xi|label))
        p_label = sorted(p_label.items(), key=lambda x: x[1], reverse=True)
        return p_label[0][0]

def test_run():
    ctc = ChineseTextClassification(initial=0)
    test_set = ctc.test_set
    ctc = ChineseTextClassification()
    right = 0
    wrong = 0
    for test_case, label in test_set:
        if ctc.predict(pickle.loads(test_case)) == label:
            right += 1
        else:
            wrong += 1
    print(float(right) / (right + wrong))

test_run()
