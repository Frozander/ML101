import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x / (1 - x)

def relu(x):
    return x*(x>0) #Python can implicitly convert boolean to number

def relu_der(x):
    return 1*(x>0)

def leaky_relu(x):
    if x < 0:
        return 0.01 * x
    return x

def leaky_relu_der(x):
    if x < 0:
        return 0.01
    return 1

def swish(x):
    return x / (1 + np.exp(-x))

def swish_der(x):
    return np.exp(x) * (np.exp(x) + x + 1) / ((np.exp(x) + 1)**2)

def identity(x):
    return x