import numpy as np
import matplotlib.pyplot as plt
from

x = np.linspace(-1, 1, 100)
y = 0.1 * x + np.power(x, 2) + np.power(x, 3)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# hyperparameters
LEARNING_RATE = 1e-2
N_NEURONS = 500
EPOCHS = 5000

# parameter initialization
random_state = np.random.RandomState(42)

# the following shapes are better for broadcasting in NumPy
w1 = random_state.randn(1, N_NEURONS)
b1 = np.zeros(shape=(1, N_NEURONS))
w2 = random_state.randn(1, N_NEURONS)
b2 = 0.

# training
loss_collection = []
y_hat_collection = []
for e in range(EPOCHS):
    # forwardpass
    y_hat = forward_pass(x, w1, b1, w2, b2)
    loss = np.mean(mse(y, y_hat))  # the loss over the entire training data
    loss_collection.append(loss)

    if (e + 1) % 500 == 0:
        print('[epoch %4d] [loss: %.4f]' % (e + 1, loss))
        y_hat_collection.append(y_hat)

    # calculate gradient
    dLdw1, dLdb1, dLdw2, dLdb2 = cal_gradient(x, y, y_hat, w1, b1, w2, b2)

    # calculate gradient over entire dataset
    dLdw1 = np.mean(dLdw1, axis=0, keepdims=True)
    dLdb1 = np.mean(dLdb1, axis=0, keepdims=True)
    dLdw2 = np.mean(dLdw2, axis=0, keepdims=True)
    dLdb2 = np.mean(dLdb2, axis=0)

    # update parameters
    w1 = update_parameters(w1, dLdw1, LEARNING_RATE)
    b1 = update_parameters(b1, dLdb1, LEARNING_RATE)
    w2 = update_parameters(w2, dLdw2, LEARNING_RATE)
    b2 = update_parameters(b2, dLdb2, LEARNING_RATE)


# plot the results
y_pred = forward_pass(x, w1, b1, w2, b2)
plt.figure(figsize=(5, 5))
plt.plot(x, y, label='y')
plt.plot(x, y_pred, label='y_pred')
plt.legend()
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(loss_collection)
plt.xticks(ticks=np.linspace(0, EPOCHS, 5, dtype=int),
           labels=np.linspace(0, EPOCHS, 5, dtype=int))
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()