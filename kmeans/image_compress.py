import cv2
import random
import math

class ImageCompress:
    def __init__(self, filename):
        self.image = cv2.imread(filename)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.m = self.height * self.width # 图片中的像素点数

    def get_distance(self, vec_1, vec_2):
        dist = 0.0
        for i, j in zip(vec_1, vec_2):
            dist += (float(i) - float(j)) ** 2
        return math.sqrt(dist)

    def kmeans(self, k=16, iter_times=50):
        self.cluster = [] 
        # 随机选取k个质心
        for cluster_idx in range(k):
            while True:
                i = random.randint(0, self.m - 1)
                temp_cluster = list(self.image[i // self.width][i % self.width])
                print(temp_cluster) # DEBUG
                if temp_cluster in self.cluster:
                    continue
                else:
                    self.cluster.append(temp_cluster)
                    break
        self.best_match = [[] for i in range(k)] # best_match[i]为第i个质心聚集的像素点的标号list
        for i in range(iter_times):
            print('第%d次迭代' % i + 1)
            temp_match = [[] for j in range(k)]
            for px in range(self.m):
                h = px // self.width
                w = px % self.width
                nearest_cluster = -1
                min_dist = -1
                for cluster_idx in range(k):
                    dist = self.get_distance(self.cluster[cluster_idx], self.image[h][w])
                    if cluster_idx == 0:
                        nearest_cluster = cluster_idx
                        min_dist = dist
                    else:
                        if min_dist > dist:
                            min_dist = dist
                            nearest_cluster = cluster_idx
                temp_match[nearest_cluster].append(px)
            if temp_match == self.best_match:
                break
            else:
                self.best_match = temp_match
                for cluster_idx in range(k):
                    for j in range(len(self.cluster[0])):
                        self.cluster[cluster_idx][j] = float(sum([self.image[px // self.width][px % self.width][j] for px in temp_match[cluster_idx]])) / len(temp_match[cluster_idx])

    def compress(self, filename):
        for cluster_idx in range(len(self.cluster)):
            for px in self.best_match[cluster_idx]:
                h = px // self.width
                w = px % self.height
                self.image[h][w] = self.cluster[cluster_idx][:]
        cv2.imwrite(filename, self.image)

def test_run(in_filename, out_filename):
    ic = ImageCompress(in_filename)
    ic.kmeans()
    ic.compress(out_filename)

test_run('./test_in.png', './test_out.png')
