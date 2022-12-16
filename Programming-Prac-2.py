import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid and it's derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Plot sigmoid and it's derivative
x = np.linspace(-5, 5, 100)
sigmoid_x = sigmoid(x)
d_sigmoid_x = derivative_sigmoid(x)

plt.figure(figsize=(5, 5))
plt.plot(x, sigmoid_x, label='sigmoid')
plt.plot(x, d_sigmoid_x, label='derivative of sigmoid')
plt.legend()
plt.show()

# Task 2:
# Define neural network with one hidden layer
# the parameter x can be a batch of dataset with the shape of number_of_samples x number_of_dimensions
def forward_pass(x, w1, b1, w2, b2):
    z = sigmoid(np.dot(x, w1) + b1)
    y_hat = np.dot(z, w2.transpose()) + b2
    return y_hat

# Define mean squared error loss and it's derivative
def mse(y_true, y_pred):
    # the factor 0.5 is used for easier calucation of the derivative
    return 0.5 * np.sum(np.square(y_true - y_pred), axis=-1)

def derivative_mse(y_true, y_pred):
    return y_pred - y_true


# Please write a function cal_gradient(x, y, y_hat, w1, b1, w2, b2) that returns the gradients
# w.r.t. the four learnable parameters: w1, b1, w2 and b2.
# Hint:
# y refers to the ground truth and y_hat refers to the output of a neural network;
# recall the chain rule in calculating gradient;
# first calculate all derivatives by hand and transform your results into codes;
# the shape of weights and bias might be different with mathematical formulas due to the broadcasting property of NumPy
def cal_gradient(x, y, y_hat, w1, b1, w2, b2):
    ly1 = np.dot(x, w1) + b1
    dLdw1 = derivative_mse(y, y_hat) * derivative_sigmoid(ly1) * x * w2.reshape(1, -1)
    dLdb1 = derivative_mse(y, y_hat) * derivative_sigmoid(ly1) * w2.reshape(1, -1)
    dLdw2 = derivative_mse(y, y_hat) * sigmoid(ly1)
    dLdb2 = derivative_mse(y, y_hat)
    return dLdw1, dLdb1, dLdw2, dLdb2

# Please define a function update_parameters(parameters, gradient, learning_rate) that
# returns the updated parameters by applying the standard gradient descent once.
# Hint: recall the definition of standard gradient descent algorithm without momentum and decays.
def update_parameters(parameters, gradient, learning_rate):
    return parameters - learning_rate*gradient