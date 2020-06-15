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
            self.W += np.dot(train_data, train_data.T) / self.q
        for i in range(self.size):
            self.W[i][i] = 0
        
        self.theta = 0

    def energy(self, x):
        """calculate potential energy"""
        return - np.dot(np.dot(self.W, x), x) / 2

    def update(self, data):
        updated = np.copy(data)
        for i in range(self.q):
            v0 = energy(data[i])
            while True:
                data[i] = np.sign(np.dot(self.W, data[i]) - self.theta)
                v1 = energy(data[i])
                print("now recollecting")
                if v0 == v1:
                    updated[i] = data[i]
                    break
        return updated

    def visualize(self, data):
        for i in range(self.q):
            data[i] = np.reshape(data[i], [5,5])
            plt.imshow(data[i], cmap = 'gray', vmin = -1, vmax = 1, interpolation = 'none')
            plt.show()

    def noise(self, data, ratio):
        for i in range(self.q):
            for j in range(self.size):
                if random.random() <= ratio:
                    data[i][j] = - data[i][j]
        return data

def train(train_data):
    hopfield = Hopfield(train_data)
    hopfield.visualize(train_data)
    init = hopfield.noise(train_data, 0.10)
    recollected = hopfield.update(init)
    hopfield.visualize(recollected)

if __name__ == "__main__":
    train_data = [np.array([-1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1])]
    train(train_data)