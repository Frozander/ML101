import numpy as np
import ml101.functional as ml_funcs
from ml101.perceptron import Perceptron


if __name__ == "__main__":

    # AND logic gate training
    train_values = np.array([-40, -10, 0, 8, 15, 22, 38],
                      dtype=float)
    train_outputs = np.array([-40, 14, 32, 46, 59, 72, 100],
                      dtype=float)

    my_perceptron = Perceptron(1, activation_fn=ml_funcs.identity, loss_fn=ml_funcs.relu_der)
    my_perceptron.train(train_values, train_outputs)

    ## Predition
    test1 = np.array([0])
    test2 = np.array([100])
    test3 = np.array([-40])
    test4 = np.array([5])

    print("Prediction for {}: {}".format(test1, my_perceptron.predict(test1)))
    print("Prediction for {}: {}".format(test2, my_perceptron.predict(test2)))
    print("Prediction for {}: {}".format(test3, my_perceptron.predict(test3)))
    print("Prediction for {}: {}".format(test4, my_perceptron.predict(test4)))
    print("Current weights: {}".format(my_perceptron.weights))