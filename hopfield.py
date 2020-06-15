import numpy as np
import matplotlib.pyplot as plt
import random

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

        #print(self.W)

    def energy(self, x):
        """calculate potential energy"""
        return - np.dot(np.dot(self.W, x), x) / 2 + np.sum(self.theta * x)

    def update(self, data, train_data, trial):
        sim_list = []
        acc_list = []
        for i in range(self.q):
            #v = self.energy(data[i])
            #print(v)
            sim_ave = 0
            acc_ratio = 0
            for j in range(trial):
                data[i] = np.sign(np.dot(self.W, data[i]) - self.theta)
                #print("updating")
                #print(data)

                #calculate recall performance
                sim, acc = self.performance(data[i], train_data[i])
                sim_ave += sim / trial
                acc_ratio += acc / trial
            sim_list.append(sim_ave)
            acc_list.append(acc_ratio)

            """
            while True:
                cnt += 1
                v0 = self.energy(data[i])
                data[i] = np.sign(np.dot(self.W, data[i]) - self.theta)
                v1 = self.energy(data[i])
                print("now recollecting")

                acc, ratio = self.performance(data[i], train_data[i])
                acc_list.append(acc)

                if v0 == v1:
                    break
            """
            #self.visualize(data)
        return data, sim_list, acc_list

    def visualize(self, data):
        for i in range(self.q):
            x = np.reshape(data[i], [5,5])
            plt.imshow(x, cmap = 'gray', vmin = -1, vmax = 1, interpolation = 'none')
            plt.show()

    def noise(self, data, ratio):
        for i in range(self.q):
            for j in range(self.size):
                if random.random() <= ratio:
                    #print(data[i])
                    data[i][j] = - data[i][j]
        #print("Add noise")
        return data

    def performance(self, data, train_data):
        sim = 0
        acc = 0
        for i in range(self.size):
            if data[i] == train_data[i]:
                sim += 1
            sim = sim / self.size
            
        if (data == train_data).all():
            acc += 1
        return sim, acc
        

def train(train_data):
    hopfield = Hopfield(train_data)
    hopfield.visualize(train_data)
    init = hopfield.noise(train_data, 0.10)
    hopfield.visualize(init)
    recollected , sim, acc = hopfield.update(init, train_data, 5)
    hopfield.visualize(recollected)
    print("similarity : {}, accuracy : {}".format(sim, acc))
    #print("recalled")

if __name__ == "__main__":
    train_data = np.array([np.array([-1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1]),
                            np.array([1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1]),
                            np.array([-1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1])])
    train(train_data)