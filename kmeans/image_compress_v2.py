import cv2
import random
import math
import numpy as np

class ImageCompress:
    def __init__(self, filename):
        self.image = cv2.imread(filename)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.channel = self.image.shape[2]
        self.m = self.height * self.width # 图片中的像素点数 
        self.reshape_image = self.image.reshape(self.m, self.channel)       
        self.X = np.mat(self.reshape_image, dtype=float)

    def get_distance(self, vec_1, vec_2):
        return np.linalg.norm(vec_1 - vec_2)

    def kmeans(self, k=16, iter_times=50):
        self.cluster = [] 
        # 随机选取k个质心
        for cluster_idx in range(k):
            while True:
                i = random.randint(0, self.m - 1)
                temp_cluster = list(self.reshape_image[i])
                print(temp_cluster) # DEBUG
                if temp_cluster in self.cluster:
                    continue
                else:
                    self.cluster.append(temp_cluster)
                    break
        self.cluster = np.mat(self.cluster, dtype=float)
        self.best_match = [[] for i in range(k)] # best_match[i]为第i个质心聚集的像素点的标号list
        for i in range(iter_times):
            print('第%d次迭代' % (i + 1))
            temp_match = [[] for j in range(k)]
            for px in range(self.m):
                temp_match[np.argmin(np.mean(np.multiply((self.cluster - self.X[px]), (self.cluster - self.X[px])), axis=1))].append(px)
            if temp_match == self.best_match:
                break
            else:
                self.best_match = temp_match
                for cluster_idx in range(k):
                    if temp_match[cluster_idx] == []:
                        print(cluster_idx)
                    self.cluster[cluster_idx] = np.mean(self.X[temp_match[cluster_idx],:], axis=0)[0]

    def compress(self, filename):
        for cluster_idx in range(len(self.cluster)):
            for px in self.best_match[cluster_idx]:
                self.reshape_image[px] = self.cluster[cluster_idx][:]
        cv2.imwrite(filename, self.image)

def test_run(in_filename, out_filename):
    ic = ImageCompress(in_filename)
    ic.kmeans()
    ic.compress(out_filename)

test_run('./test_in.png', './test_out.png')
