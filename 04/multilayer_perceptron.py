import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        self.net = {}
        pass

    def init_network(self):
        net = {}
        # layer 1
        net['w1'] = np.array([[0.2, 0.7, 0.4],[0.6, 0.1, 0.9]])
        net['b1'] = np.array([1, 1, 1])
        # layer 2
        net['w2'] = np.array([ [0.7, 0.2], [0.8, 0.5], [0.88, 0.9124] ])
        net['b2'] = np.array([0.5, 0.5])
        # layer 3 <-- output
        net['w3'] = np.array([ [0.27, 0.71], [0.423, 0.7314] ])
        net['b3'] = np.array([0.9, 0.2])

        self.net = net

    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)

        return y
    
    def identity(self, x):
        return x
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def step(self, x):
         return np.array(x > 0).astype(int)