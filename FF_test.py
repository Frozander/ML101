import numpy as np
import ml101.functional as ml_funcs
from ml101.FForward import FeedForward


if __name__ == "__main__":

    # AND logic gate training
    train_values = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    train_outputs = np.array([0, 1, 1, 0])

    fforward = FeedForward(input_size=np.array(1, 1),
                                hidden_size=np.array(1, 1),
                                output_size=np.array(1, 1),
                                lr=0.01,
                                activation_fn=ml_funcs.sigmoid,
                                loss_fn=ml_funcs.sigmoid_der)
    fforward(train_values, train_outputs, 10000)

    ## Predition
    test1 = np.array([0])
    test2 = np.array([100])
    test3 = np.array([-40])
    test4 = np.array([5])