import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro']


class Hopfield:
    def __init__(self, train_data):
        """ setup weight matrix W based on Hebbian"""
        self.size = len(train_data[0])
        # weight matrix
        self.W = np.zeros((self.size, self.size))
        # number of patterns
        self.q = len(train_data)
        for i in range(self.q):
            self.W += np.outer(train_data[i], train_data[i]) / self.q
        for i in range(self.size):
            self.W[i][i] = 0
        
        self.theta = 0

    def energy(self, x):
        """calculate potential energy"""
        return - np.dot(np.dot(self.W, x), x) / 2 + np.sum(self.theta * x)

    def update(self, data, train_data):
        #asynchronous type
        for _ in range(300):
            #v = self.energy(data)
            n = random.randint(0, self.size - 1)
            data[n] = np.sign(np.dot(self.W[n], data) - self.theta)
        sim, acc = self.performance(data, train_data)
    
        return data, sim, acc

    def synchro_update(self, data, train_data):
        #synchronous type
        for _ in range(300):
            data = np.sign(np.dot(self.W, data) - self.theta)
        #calculate recall performance
        sim, acc = self.performance(data[i], train_data[i])
        return data, sim, acc

    def noise(self, data, ratio):
        for i in range(self.q):
            for j in range(self.size):
                if random.random() <= ratio:
                    data[i][j] = - data[i][j]
        #print("Add noise")
        return data

    def performance(self, data, train_data):
        sim = 0
        acc = 0
        sim = np.dot(data.T, train_data) / self.size
        
        if (data == train_data).all():
            acc += 1
        return sim, acc
        
def visualize(data, idx):
        x = np.reshape(data, [5,5])
        plt.imshow(x, cmap = 'gray', vmin = -1, vmax = 1, interpolation = 'none')
        #plt.title("image {}".format(idx+1))
        plt.show()

def train(train_data, noise, idx):
    original_data = np.copy(train_data)
    hopfield = Hopfield(train_data)
    init = hopfield.noise(train_data, noise)
    recollected , sim, acc = hopfield.update(init[idx], original_data[idx])
    return sim, acc

def syn_train(train_data, noise, idx):
    original_data = np.copy(train_data)
    hopfield = Hopfield(train_data)
    init = hopfield.noise(train_data, noise)
    recollected , sim, acc = hopfield.update(init[idx], original_data[idx])
    return sim, acc

if __name__ == "__main__":
    train_data1 = np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1])])
    train_data2 = np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1])])
    train_data3 = np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1])])
    train_data4 =  np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1]),
                            np.array([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1])])
    train_data5 = np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1]),
                            np.array([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1]),
                            np.array([-1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1])])
    train_data6 = np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1]),
                            np.array([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1]),
                            np.array([-1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1]),
                            np.array([-1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1])])
    total_sim = 0
    total_acc = 0
    
    #for i in range(6):
    #    visualize(train_data6[i], i)
    
    sim_list = []
    acc_list = []
    #for the experiment of performance using 6 kinds of data
    input_data = [train_data1, train_data2, train_data3, train_data4, train_data5, train_data6]
    for i in range(6):
        for j in range(1000):
            sim, acc = train(input_data[i], 0.10, random.randint(0, i))
            total_sim += sim
            total_acc += acc
        total_sim /= 1000
        total_acc /= 1000
        #print("similarity : {}, accuracy : {}".format(total_sim, total_acc))
        sim_list.append(total_sim)
        acc_list.append(total_acc)
    
    print(sim_list, acc_list)
    x = [1, 2, 3, 4, 5, 6]
    plt.title("元画像に対する想起性能")
    plt.xlabel("記憶パターン数（個）")
    #plt.ylabel("正解との類似度の全試行平均")
    plt.ylabel("正答率（元画像を完全再現した頻度割合）")
    #plt.plot(x, sim_list)
    plt.plot(x, acc_list)
    plt.show()
    
    #for the experiment of noise using data2 and data4
    sim_list = []
    acc_list = []
    noise_list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    data = [train_data2, train_data4]
    for i in range(2):
        for j in noise_list:
            for k in range(1000):
                sim, acc = train(data[i], j, random.randint(0,(i+1)*2-1))
                total_sim += sim
                total_acc += acc
            total_sim /= 1000
            total_acc /= 1000
            sim_list.append(total_sim)
            acc_list.append(total_acc)
    print(sim_list, acc_list)
    noise_list = [5, 10, 15, 20, 30 ,50, 75, 100]
    plt.title("ノイズを変化させた時の想起性能")
    plt.xlabel("ノイズ（%）")
    plt.ylabel("正解との類似度の全試行平均")
    #plt.ylabel("正答率（元画像を完全再現した頻度割合）")
    plt.plot(noise_list, sim_list[:8], label = "2種類")
    plt.plot(noise_list, sim_list[8:], label = "4種類")
    #plt.plot(noise_list, acc_list[:8], label = "2種類")
    #plt.plot(noise_list, acc_list[8:], label = "4種類")
    plt.legend()
    plt.show()

    """
    #Compare asynchronous type with synchronous type
    total_asyn_sim = 0
    total_asyn_acc = 0
    total_syn_sim = 0
    total_syn_acc = 0
    syn_sim_list = []
    syn_acc_list = []
    asyn_sim_list = []
    asyn_acc_list = []

    #for the experiment of performance using 6 kinds of data
    input_data = [train_data1, train_data2, train_data3, train_data4, train_data5, train_data6]
    for i in range(6):
        for j in range(1000):
            asyn_sim, asyn_acc = train(input_data[i], 0.10, random.randint(0, i))
            syn_sim, syn_acc = syn_train(input_data[i], 0.10, random.randint(0, i))
            total_asyn_sim += asyn_sim
            total_asyn_acc += asyn_acc
            total_syn_sim += syn_sim
            total_syn_acc += syn_acc
        total_asyn_sim /= 1000
        total_asyn_acc /= 1000
        total_syn_sim /= 1000
        total_syn_acc /= 1000
        #print("similarity : {}, accuracy : {}".format(total_sim, total_acc))
        asyn_sim_list.append(total_asyn_sim)
        asyn_acc_list.append(total_asyn_acc)
        syn_sim_list.append(total_syn_sim)
        syn_acc_list.append(total_syn_acc)
    
    #print(sim_list, acc_list)
    x = [1, 2, 3, 4, 5, 6]
    plt.title("元画像に対する想起性能")
    plt.xlabel("記憶パターン数（個）")
    #plt.ylabel("正解との類似度の全試行平均")
    plt.ylabel("正答率（元画像を完全再現した頻度割合）")
    #plt.plot(x, sim_list)
    plt.plot(x, asyn_acc_list, label = "非同期更新")
    plt.plot(x, syn_acc_list, label = "同期更新")
    plt.legend()
    plt.show()
    
    #for the experiment of noise using data2 and data4
    noise_list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    data = [train_data2, train_data4]
    for i in range(2):
        for j in noise_list:
            for k in range(1000):
                asyn_sim, asyn_acc = train(data[i], j, random.randint(0,(i+1)*2-1))
                syn_sim, syn_acc = syn_train(data[i], j, random.randint(0,(i+1)*2-1))
                total_asyn_sim += asyn_sim
                total_asyn_acc += asyn_acc
                total_syn_sim += syn_sim
                total_syn_acc += syn_acc
            total_asyn_sim /= 1000
            total_asyn_acc /= 1000
            total_syn_sim /= 1000
            total_syn_acc /= 1000
            asyn_sim_list.append(total_asyn_sim)
            asyn_acc_list.append(total_asyn_acc)
            syn_sim_list.append(total_syn_sim)
            syn_acc_list.append(total_syn_acc)
    #print(sim_list, acc_list)
    noise_list = [5, 10, 15, 20, 30 ,50, 75, 100]
    plt.title("ノイズを変化させた時の想起性能")
    plt.xlabel("ノイズ（%）")
    #plt.ylabel("正解との類似度の全試行平均")
    plt.ylabel("正答率（元画像を完全再現した頻度割合）")
    plt.plot(noise_list, asyn_acc_list[:8], label = "非同期2種類")
    plt.plot(noise_list, asyn_acc_list[8:], label = "非同期4種類")
    plt.plot(noise_list, syn_acc_list[:8], label = "同期2種類")
    plt.plot(noise_list, syn_acc_list[8:], label = "同期4種類")
    #plt.plot(noise_list, acc_list[:8], label = "2種類")
    #plt.plot(noise_list, acc_list[8:], label = "4種類")
    plt.legend()
    plt.show()
    """