import numpy as np
import ml101.activations as ml_funcs
from ml101.perceptron import Perceptron


if __name__ == "__main__":

    # AND logic gate training
    train_values = np.array([[1,1], [1, 0], [0, 1], [0, 0]])
    train_outputs = np.array([1, 0, 0, 0])

    my_perceptron = Perceptron(2)
    my_perceptron.train(train_values, train_outputs)

    ## Predition
    test1 = np.array([1, 1])
    test2 = np.array([1, 0])
    test3 = np.array([0, 1])
    test4 = np.array([0, 0])

    print("Prediction for {}: {}".format(test1, my_perceptron.predict(test1)))
    print("Prediction for {}: {}".format(test2, my_perceptron.predict(test2)))
    print("Prediction for {}: {}".format(test3, my_perceptron.predict(test3)))
    print("Prediction for {}: {}".format(test4, my_perceptron.predict(test4)))
    print("Current weights: {}".format(my_perceptron.weights))