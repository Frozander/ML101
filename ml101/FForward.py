import numpy as np
from .functional import sigmoid, sigmoid_der

class FeedForward():
    
    def __init__(self, input_size, hidden_size=3, output_size=1, lr=0.01, activation_fn=sigmoid, loss_fn=sigmoid_der):
        self.input_size = input_size
        self.output_size = output_size
        self.synaptic_weights_in = 2 * np.random.rand(self.input_size.shape[0], self.hidden_size.shape[1]) - 1
        self.synaptic_weights_out = 2 * np.random.rand(self.hidden_size.shape[0], self.output_size.shape[1]) - 1
        self.bias = np.random.random(1)
        self.lr = lr
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
    
    def __call__(self, input_data, output_data, epoch=100):
        for _ in range(epoch):
            for data, label in zip(input_data, output_data):
                self.forward(input_data)
                self.backward(input_data, output_data)
    
    def forward(self, x):
        self.hidden_calc = self.activation_fn(np.dot(x, self.synaptic_weights_in))
        self.output_calc = self.activation_fn(np.dot(self.hidden_calc, self.synaptic_weights_out))
    
    def backward(self, x, y):
        self.synaptic_weights_out += np.dot(self.hidden_calc.T, self.lr * (y - self.output_calc) * self.loss_fn(self.output_calc))
        self.synaptic_weights_in += np.dot(x.T, (np.dot(lr * (y - self.output_calc) * self.loss_fn(self.output_calc), self.synaptic_weights_out.T * self.loss_fn(self.hidden_calc))))