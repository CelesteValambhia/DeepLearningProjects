import numpy as np
import matplotlib.pyplot as plt

# Creating 1D dataset using numpy
n_samples = 100
X = np.linspace(-1, 1, n_samples)  # Evenly spaced 100 samples between (-1,1)
Y = 0.1 * X + np.power(X, 2) + np.power(X, 3)  # or Y = 0.1*X + X**2 + X**3

# Cresting function
def create_toy_dataset(n_samples=100):
    X = np.linspace(-1, 1, n_samples)   # Evenly spaced 100 samples between (-1,1)
    Y = 0.1 * X + np.power(X, 2) + np.power(X, 3)   # or Y = 0.1*X + X**2 + X**3
    return X, Y

# Creating Class which returns dataset for user-defined samples
class Dataset():
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def load_data(self):
        self.X = np.linspace(-1, 1, self.n_samples)
        self.Y = 0.1 * self.X + np.power(self.X, 2) + np.power(self.X, 3)
        return self.X, self.Y

# Plot the 1D dataset above. The figure size should be set to 5 x 5. The x-axis should only consist of five ticks from -1 to +1.
plt.figure(figsize=(5, 5))
plt.plot(X, Y)
plt.xticks(ticks=np.linspace(-1, 1, 5), labels=np.linspace(-1, 1, 5))
plt.show()


# Assignment 2: Construct 2D dataset where each dimention of x is sampled from standard normal distribution and
# Y=0 if ||x^2||<1 else Y=1

# random_state = np.random.RandomState(0)
# X = random_state.randn(100, 2)
np.random.seed(0)
X = np.random.randn(100, 2)
Y = np.ones(shape=(100, ))
Y[np.sum(np.square(X), axis=-1) < 1] = 0

# Please plot the 2D dataset above. The figure size should be set to 5 x 5. The samples affiliated
# to class 1 should be red triangles, while the rest samples are blue circles. Add one legend to the
# figure to indicate the two classes.
plt.figure(figsize=(5, 5))
plt.scatter(X[Y==0, 0], X[Y==0, 1], marker='o', c='b', label='class 0')
plt.scatter(X[Y==1, 0], X[Y==1, 1], marker='v', c='r', label='class 1')
plt.legend()
plt.show()

# Assignment 3
X = np.arange(0, 16).reshape(4, 4)
W = np.ones(shape=(2, 2))

# Calculate the convolution output (stride=1, dilation=1, no padding) using NumPy. Subsequently, write a
# function convolve(X, W) to calculate any convolution regarding the given X and the filter W (still:
# stride=1, dilation=1, no padding).
X_conv = np.zeros(shape=(3, 3))     # initializing 3x3 matrix
for i in range(X_conv.shape[0]):
    for j in range(X_conv.shape[1]):
        X_conv[i, j] = np.sum(X[i:(i+2), j:(j+2)] * W)
print(X_conv)


def convolve(X, W):
    (X_height, X_width) = X.shape
    (W_height, W_width) = W.shape

    # X_height and X_width should be larger than W_height and W_width, respectively.

    X_conv_height, X_conv_width = X_height - W_height + 1, X_width - W_width + 1
    X_conv = np.zeros(shape=(X_conv_height, X_conv_width))

    for i in range(X_conv.shape[0]):
        for j in range(X_conv.shape[1]):
            X_conv[i, j] = np.sum(X[i:(i + W_height), j:(j + W_width)] * W)
    return X_conv