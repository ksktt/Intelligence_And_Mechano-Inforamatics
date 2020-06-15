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
            self.W += np.dot(train_data[i], train_data[i].T) / self.q
        for i in range(self.size):
            self.W[i][i] = 0
        
        self.theta = 0

    def energy(self, x):
        """calculate potential energy"""
        return - np.dot(np.dot(self.W, x), x) / 2 + np.sum(self.theta * x)

    def update(self, data):
        for i in range(self.q):
            v0 = self.energy(data[i])
            print(v0)
            for j in range(10):
                data[i] = np.sign(np.dot(self.W, data[i]) - self.theta)
                print("updating")
                print(data)
                #self.visualize(data)
            """
            while True:
                data[i] = np.sign(np.dot(self.W, data[i]) - self.theta)
                v1 = energy(data[i])
                print("now recollecting")
                if v0 == v1:
                    updated[i] = data[i]
                    break
            """
        return data

    def visualize(self, data):
        for i in range(self.q):
            x = np.reshape(data[i], [5,5])
            plt.imshow(x, cmap = 'gray', vmin = -1, vmax = 1, interpolation = 'none')
            plt.show()

    def noise(self, data, ratio):
        for i in range(self.q):
            for j in range(self.size):
                if random.random() <= ratio:
                    print(data[i])
                    data[i][j] = - data[i][j]
        print("Add noise")
        return data

def train(train_data):
    hopfield = Hopfield(train_data)
    hopfield.visualize(train_data)
    init = hopfield.noise(train_data, 0.10)
    recollected = hopfield.update(init)
    hopfield.visualize(recollected)
    print("recollected!")

if __name__ == "__main__":
    train_data = np.array([np.array([-1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1])])
    train(train_data)