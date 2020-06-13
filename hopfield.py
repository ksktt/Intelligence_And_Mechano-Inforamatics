import numpy as np

class Hopfield:
    def __init__(self, train_data):
        """ setup weight matrix W based on Hebbian"""
        self.size = len(train_data[0])
        # weight matrix
        self.W = np.zeros((self.size, self.size))
        # number of patterns
        q = len(train_data)
        for i in range(q):
            self.W += np.dot(train_data, train_data.T) / q
        for i in range(self.size):
            self.W[i][i] = 0
        
        self.theta = 0

    def energy(self, x):
        """calculate potential energy"""
        return - np.dot(np.dot(self.W, x), x) / 2

    def update(self, x):
        v0 = energy(x)
        while True:
            x = np.sign(np.dot(self.W, x) - self.theta)
            v1 = energy(x)
            if v0 == v1:
                return x
                break

def train(train_data):
    

if __name__ == "__main__":

