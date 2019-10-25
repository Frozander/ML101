import numpy as np
from .activations import *

class Perceptron():
    
    def __init__(self, input_number, threshold=10000, lr=0.01, bias=0):
        self.threshold = threshold
        self.lr = lr
        self.weights = np.zeros(input_number + 1)
        self.weights[0] = bias
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_der(self, x):
        return x / (1 - x)
    
    def predict(self, inputs):
        total = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.sigmoid(total)
        
    def train(self, inputs, labels):        
        for _ in range(self.threshold):
            for value, label in zip(inputs, labels):
                
                prediction = self.predict(value)
                
                error = label - prediction
                loss = self.sigmoid_der(error)
                
                self.weights[1:] += self.lr * loss * value
                self.weights[0] += self.lr * loss