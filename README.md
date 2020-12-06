# ML101
Repository for my [ML101 series on my own blog](https://blog.frozander.xyz/machine-learning-101-basics-and-first-perceptron/)

## What is this about?
This project is for me to relearn the basic concepts of machine learning and try to share my knowledge t ocomplete beginners who want to learn what is under the hood (more or less) of modern frameworks like Tensorflow, Pytorch, Caffe etc.
During this project I will try to build a somewhat usable machine learning framework alongside the followers of the article series (if there are any).

# What's inside?
## Perceptron
This is the simplest a neural network can get. It is a single unit with variable activation and loss function.
```python
# An example perceptron usage
my_perceptron = Perceptron(1,
                           threshold=10000,
                           lr=0.01,
                           bias=0,
                           activation_fn=sigmoid,
                           loss_fn=sigmoid_der)
my_perceptron.train(train_input, train_output)
my_perceptron.predict(test_input)
```
