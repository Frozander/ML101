import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x / (1 - x)

def relu(x):
    return x*(x>0) #Python can implicitly convert boolean to number

def relu_der(x):
    return 1*(x>0)