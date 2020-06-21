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

    def update(self, data, train_data):
        v0 = self.energy(data)
        """
        #synchronous type
        for j in range(trial):
            data = np.sign(np.dot(self.W, data) - self.theta)
        #calculate recall performance
        sim, acc = self.performance(data[i], train_data[i])
        """

        #asynchronous type
        for _ in range(300):
            #v = self.energy(data)
            n = random.randint(0, self.size - 1)
            data[n] = np.sign(np.dot(self.W[n], data) - self.theta)
        sim, acc = self.performance(data, train_data)
        
        """
        #asynchronous type
        while True:
            print(v0)
            n = random.randint(0, self.size - 1)
            data[n] = np.sign(np.dot(self.W[n], data) - self.theta)
            v1 = self.energy(data)
            print(v1)
            if v0 == v1:
                break

            v0 = v1
            
        sim, acc = self.performance(data, train_data)
        """
        return data, sim, acc

    def visualize(self, data):
        x = np.reshape(data, [5,5])
        plt.imshow(x, cmap = 'gray', vmin = -1, vmax = 1, interpolation = 'none')
        plt.show()

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
        for i in range(self.size):
            if data[i] == train_data[i]:
                sim += 1
        sim = sim / self.size
            
        if (data == train_data).all():
            acc += 1
        return sim, acc
        

def train(train_data, noise, idx):
    original_data = np.copy(train_data)
    hopfield = Hopfield(train_data)
    #hopfield.visualize(train_data[idx])
    init = hopfield.noise(train_data, noise)
    #hopfield.visualize(init[idx])
    recollected , sim, acc = hopfield.update(init[idx], original_data[idx])
    return sim, acc
    #hopfield.visualize(recollected)

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
                            np.array([-1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1])])
    train_data6 = np.array([np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]),
                            np.array([1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1]),
                            np.array([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1]),
                            np.array([-1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1]),
                            np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1])])
    total_sim = 0
    total_acc = 0
    for i in range(1000):
        sim, acc = train(train_data3, 0.10, 0)
        total_sim += sim
        total_acc += acc
    total_sim /= 1000
    total_acc /= 1000
    print("similarity : {}, accuracy : {}".format(total_sim, total_acc))
#optimize theta