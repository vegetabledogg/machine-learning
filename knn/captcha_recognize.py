import cv2
import os

class CaptchaRecognize:
    def __init__(self, file_path='./train-set'):
        self.train_set = []
        for i in range(10):
            for j in range(6):
                filename = os.path.join(file_path, str(i) + str(j) + '.png')
                image = cv2.imread(filename, 0)
                self.train_set.append([image, i]) # 将[图片BGR值list, label]存入训练集

    def get_distance(self, image_1, image_2):
        height = min(image_1.shape[0], image_2.shape[0])
        width = min(image_1.shape[1], image_2.shape[1])
        dist = 0
        for i in range(height):
            for j in range(width):
                if image_1[i, j] != image_2[i, j]: # 对应像素点不同，距离增加1，因为图片已经经过二值化处理，所以可以直接这样计算
                    dist += 1
        return dist

    def split_image(self, filename):
        split_image_list = []
        image = cv2.imread(filename, 0)
        ret, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY) # 图片二值化
        height = image.shape[0]
        width = image.shape[1]
        # 将图片分隔为只包含单个数字的图片
        split_image_list.append(image[0:height, 5:13])
        split_image_list.append(image[0:height, 14:22])
        split_image_list.append(image[0:height, 23:31])
        split_image_list.append(image[0:height, 32:41])
        return split_image_list

    def knn(self, k, filename):
        split_image_list = self.split_image(filename)
        result = 0
        for image in split_image_list:
            dist_label_list = []
            for train_image in self.train_set:
                dist_label_list.append([self.get_distance(image, train_image[0]), train_image[1]]) # 将[图片距离, label]存入dist_label_list
            dist_label_list = sorted(dist_label_list, key=lambda x: x[0]) # 将dist_label_list按图片距离从小到大排序
            label_dict = {}
            for i in range(k): # 统计距离最近的k个图片中每种label的数量
                label_dict.setdefault(dist_label_list[i][1], 0)
                label_dict[dist_label_list[i][1]] += 1
            label_dict = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
            result = result * 10 + label_dict[0][0]
        return result
    
def test_run(filename):
    cr = CaptchaRecognize()
    print(cr.knn(6, filename))

test_run('./test-set/test1.png')
test_run('./test-set/test2.png')
test_run('./test-set/test3.png')
test_run('./test-set/test4.png')
