import mnist_data
import numpy as np
import pickle
from layers import Relu 
from layers import Affine

class Mnist():
    def __init__(self):
        self.data = mnist_data.MnistData()
        self.params = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a)
    
    def load(self):
        (x_train, y_train), (x_test, y_test) = self.data.load()
        return (x_train, y_train), (x_test, y_test)
    
    
    def init_network(self):
        with open('bhatia_mnist_model.pkl', 'rb') as f:
            self.params = pickle.load(f)
        # Verify that the model has the required keys
        assert all(key in self.params for key in ['w1', 'b1', 'w2', 'b2']), "Model parameters are missing."

    def predict(self, x):
        W1, W2 = self.params['w1'], self.params['w2']
        B1, B2 = self.params['b1'], self.params['b2']

        # Layer 1
        a1 = np.dot(x, W1) + B1
        z1 = self.sigmoid(a1)

        # Layer 2 (Output layer)
        a2 = np.dot(z1, W2) + B2
        y = self.softmax(a2)

        return y
